"""App/workflow coherence factor: weakly rewards useful app clustering.

Higher is better. This is not an app-pure-layer rule. Apps are only one signal
for workflow discovery; layers may share app focus while containing different
shortcut sets when usage sequences and workflow windows support that split.
"""
import numpy as np
from collections import defaultdict
from core import Layout
from fitness import RewardFactor

APP_ALIASES = {
    "browser": ("msedge", "chrome", "firefox", "brave", "vivaldi"),
    "chrome": ("msedge", "chrome", "firefox", "brave", "vivaldi"),
    "edge": ("msedge", "chrome", "firefox", "brave", "vivaldi"),
    "visual studio code": ("code", "cursor"),
    "vs code": ("code", "cursor"),
    "vscode": ("code", "cursor"),
    "microsoft teams": ("teams", "ms-teams"),
    "teams": ("teams", "ms-teams"),
    "microsoft excel": ("excel",),
    "excel": ("excel",),
    "file explorer": ("explorer",),
    "explorer": ("explorer",),
    "terminal": ("windowsterminal", "pwsh", "powershell", "cmd"),
    "windows terminal": ("windowsterminal", "pwsh", "powershell", "cmd"),
    "claude": ("claude",),
    "codex": ("codex",),
}


class AppCoherenceFactor(RewardFactor):
    """Rewards app clustering as a weak workflow prior, not as app purity."""
    name = "app_coherence"
    
    def __init__(self, min_cluster_size: int = 2, weight: float = 10.0):
        self.min_cluster_size = min_cluster_size
        self.weight = weight
    
    def compute(self, layout: Layout) -> float:
        # Group assigned shortcuts by app and layer
        app_layer_counts = defaultdict(lambda: defaultdict(int))
        app_total = defaultdict(int)
        
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            sc = layout.shortcuts[sid]
            layer = layout.positions[i].layer
            app_layer_counts[sc.app][layer] += 1
            app_total[sc.app] += 1
        
        score = 0.0
        for app, layer_counts in app_layer_counts.items():
            total = app_total[app]
            if total < self.min_cluster_size:
                continue
            
            # Find the dominant layer for this app
            max_layer_count = max(layer_counts.values())
            dominant_layer = max(layer_counts, key=layer_counts.get)
            
            # Coherence = fraction of app's shortcuts on dominant layer
            coherence = max_layer_count / total
            
            # Reward apps with high coherence
            # Importance-weighted: apps with more shortcuts are more important to cluster.
            # Usage-weighted: apps the logger sees more often should matter more.
            usage_weight = self._usage_weight(layout, app)
            score += coherence * self.weight * np.log1p(total) * usage_weight
        
        return score

    def _usage_weight(self, layout: Layout, app: str) -> float:
        usage = layout.usage_data
        seconds_by_app = getattr(usage, "app_time_seconds", None) or {}
        if not seconds_by_app and usage.by_app:
            seconds_by_app = {
                name: data.get("total", 0) if isinstance(data, dict) else data
                for name, data in usage.by_app.items()
            }
        if not seconds_by_app:
            return 1.0

        normalized_seconds = {
            self._normalize_name(name): float(seconds)
            for name, seconds in seconds_by_app.items()
        }
        max_seconds = max(normalized_seconds.values()) if normalized_seconds else 0.0
        if max_seconds <= 0:
            return 1.0

        app_norm = self._normalize_name(app)
        aliases = {app_norm}
        for key, values in APP_ALIASES.items():
            if key in app_norm or app_norm in key:
                aliases.update(self._normalize_name(value) for value in values)

        seconds = 0.0
        for logged_name, logged_seconds in normalized_seconds.items():
            if logged_name in aliases or any(alias in logged_name for alias in aliases):
                seconds += logged_seconds

        if seconds <= 0:
            return 0.25
        return 0.5 + 1.5 * (np.log1p(seconds) / np.log1p(max_seconds))

    @staticmethod
    def _normalize_name(name: str) -> str:
        lowered = (name or "").lower()
        for suffix in (".exe", " (chrome/edge)"):
            lowered = lowered.replace(suffix, "")
        return "".join(ch for ch in lowered if ch.isalnum() or ch.isspace()).strip()
