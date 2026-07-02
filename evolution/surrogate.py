"""Neural surrogate model for fast fitness approximation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutSurrogate(nn.Module):
    """Predicts fitness scores from layout permutation."""
    
    def __init__(
        self,
        n_positions: int,
        n_shortcuts: int,
        n_factors: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.n_shortcuts = n_shortcuts
        self.n_factors = n_factors
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(n_shortcuts + 1, embedding_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(n_positions * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.head = nn.Linear(hidden_dim // 2, n_factors)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, layouts: torch.Tensor) -> torch.Tensor:
        x = layouts.long() + 1   # convert int32→int64 on GPU (free); embedding requires long
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return self.head(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SurrogateTrainer:
    """Trains the surrogate on exact fitness evaluations."""

    def __init__(
        self,
        surrogate: LayoutSurrogate,
        device: str = None,
        mixed_precision: bool = True,
        compile_model: bool = False,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_positions = surrogate.n_positions
        self.n_shortcuts = surrogate.n_shortcuts
        self.n_factors = surrogate.n_factors
        self.embedding_dim = surrogate.embedding_dim
        self.surrogate = surrogate.to(device)
        self.device = device
        # AMP only helps on SM 7.0+ (Volta/tensor cores). GTX 1070 = SM 6.1:
        # no tensor cores, no throughput gain, and no GradScaler → gradient underflow risk.
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        self.use_amp = bool(mixed_precision and str(device).startswith("cuda") and cap[0] >= 7)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = []
        self.mean = None
        self.std = None
        self._predict_buffer = None
        self._compiled = False
        if compile_model and hasattr(torch, "compile") and self._can_compile():
            try:
                self.surrogate = torch.compile(self.surrogate, mode="max-autotune")
                self._compiled = True
            except Exception as exc:
                print(f"  torch.compile failed, continuing eager: {exc}", flush=True)
        self.optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=1e-3)
        self.surrogate.eval()

    @staticmethod
    def _can_compile() -> bool:
        """Triton requires CUDA capability >= 7.0 (e.g., RTX 20-series+)."""
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 7

    def train(self, layouts: np.ndarray, exact_scores: np.ndarray, epochs: int = 100, batch_size: int = 256):
        assert layouts.shape[0] == exact_scores.shape[0]

        self.mean = exact_scores.mean(axis=0)
        self.std = exact_scores.std(axis=0) + 1e-6
        normalized = (exact_scores - self.mean) / self.std

        X = torch.tensor(layouts, dtype=torch.long, device=self.device)
        Y = torch.tensor(normalized, dtype=torch.float32, device=self.device)

        n = len(X)
        batch_size = min(int(batch_size), n)

        self.surrogate.train()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=1e-5
        )
        for epoch in range(epochs):
            total_loss = 0.0
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                batch_x = X.index_select(0, idx)
                batch_y = Y.index_select(0, idx)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pred = self.surrogate(batch_x)
                    # Huber loss: robust to fitness outliers (violations span many orders of magnitude)
                    loss = F.huber_loss(pred.float(), batch_y, delta=1.0)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item() * batch_x.size(0)
            scheduler.step()

            avg_loss = total_loss / n
            self.history.append(avg_loss)
            if epoch % 10 == 0:
                print(f"  Surrogate epoch {epoch}: loss={avg_loss:.6f}")
        # Keep model in eval mode between retrains — predict() does not flip it
        self.surrogate.eval()

    def predict(self, layouts: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Trainer has not been trained yet")

        # Use int32 for H2D transfer — halves bandwidth vs int64, free GPU cast in forward().
        layouts = np.asarray(layouts, dtype=np.int32)
        n = layouts.shape[0]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            if str(self.device).startswith("cuda"):
                if self._predict_buffer is None or self._predict_buffer.shape[0] < n or self._predict_buffer.shape[1] != layouts.shape[1]:
                    self._predict_buffer = torch.empty((n, layouts.shape[1]), dtype=torch.int32, device=self.device)
                cpu_view = torch.from_numpy(layouts)
                self._predict_buffer[:n].copy_(cpu_view, non_blocking=True)
                X = self._predict_buffer[:n]
            else:
                X = torch.from_numpy(layouts).to(self.device)
            pred = self.surrogate(X).float().cpu().numpy()
        return pred * self.std + self.mean
    
    def evaluate(self, layouts: np.ndarray, exact_scores: np.ndarray) -> dict:
        pred = self.predict(layouts)
        mse = np.mean((pred - exact_scores) ** 2, axis=0)
        mae = np.mean(np.abs(pred - exact_scores), axis=0)
        r2 = 1 - np.sum((pred - exact_scores) ** 2, axis=0) / (np.sum((exact_scores - exact_scores.mean(axis=0)) ** 2) + 1e-6)
        return {"mse": mse, "mae": mae, "r2": r2}
    
    def save(self, path: str):
        torch.save({
            "model": self.surrogate.state_dict(),
            "mean": self.mean,
            "std": self.std,
            "history": self.history,
            "n_positions": self.n_positions,
            "n_shortcuts": self.n_shortcuts,
            "n_factors": self.n_factors,
            "embedding_dim": self.embedding_dim,
        }, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.surrogate.load_state_dict(ckpt["model"])
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.history = ckpt.get("history", [])


class SurrogateManager:
    """Manages the surrogate during evolution."""
    
    def __init__(self, surrogate: LayoutSurrogate, trainer: SurrogateTrainer,
                 retrain_every: int = 200, exact_eval_every: int = 50,
                 retrain_epochs: int = 10, retrain_batch_size: int = 1024,
                 max_retrain_samples: int = 20000):
        self.surrogate = surrogate
        self.trainer = trainer
        self.retrain_every = retrain_every
        self.exact_eval_every = exact_eval_every
        self.retrain_epochs = int(retrain_epochs)
        self.retrain_batch_size = int(retrain_batch_size)
        self.max_retrain_samples = int(max_retrain_samples)
        self.generation = 0
        self.exact_cache = []
        self.accuracy_history = []
    
    def should_retrain(self) -> bool:
        return self.generation > 0 and self.generation % self.retrain_every == 0
    
    def should_exact_eval(self) -> bool:
        return self.generation % self.exact_eval_every == 0
    
    def add_exact_evaluations(self, layouts: np.ndarray, exact_scores: np.ndarray):
        for i in range(len(layouts)):
            self.exact_cache.append((layouts[i].copy(), exact_scores[i].copy()))
    
    def retrain(self):
        if len(self.exact_cache) < 100:
            print(f"  Surrogate cache too small ({len(self.exact_cache)}), skipping retrain")
            return

        n_cache = len(self.exact_cache)
        if self.max_retrain_samples > 0 and n_cache > self.max_retrain_samples:
            # Keep the most recent half so the surrogate follows the current
            # search basin, and fill the rest with random historical samples
            # to avoid catastrophic forgetting.
            recent_n = self.max_retrain_samples // 2
            random_n = self.max_retrain_samples - recent_n
            recent_idx = np.arange(n_cache - recent_n, n_cache)
            history_limit = n_cache - recent_n
            if history_limit > 0 and random_n > 0:
                random_idx = np.random.choice(history_limit, size=random_n, replace=False)
                indices = np.concatenate([random_idx, recent_idx])
            else:
                indices = recent_idx
        else:
            indices = np.arange(n_cache)

        layouts = np.array([self.exact_cache[int(i)][0] for i in indices])
        scores = np.array([self.exact_cache[int(i)][1] for i in indices])

        print(
            f"  Retraining surrogate on {len(layouts)} of {n_cache} exact evaluations...",
            flush=True,
        )
        self.trainer.train(
            layouts,
            scores,
            epochs=self.retrain_epochs,
            batch_size=self.retrain_batch_size,
        )

        acc = self.trainer.evaluate(layouts[:min(500, len(layouts))], scores[:min(500, len(scores))])
        r2_mean = float(np.mean(acc["r2"]))
        self.accuracy_history.append(r2_mean)
        print(f"  Surrogate R^2 = {r2_mean:.4f}", flush=True)
    
    def step(self):
        self.generation += 1
