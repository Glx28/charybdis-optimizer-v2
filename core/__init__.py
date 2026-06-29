"""Core data structures for the Charybdis keyboard layout optimizer."""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Position:
    """A physical key position on the keyboard."""
    gene_idx: int
    layer: int
    x: float
    y: float
    hand: str  # "left" or "right"
    finger: int  # 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    effort: float
    is_thumb: bool = False
    is_frozen: bool = False
    row: int = 0
    col: int = 0

    @property
    def is_left(self) -> bool:
        return self.hand == "left"

    @property
    def is_right(self) -> bool:
        return self.hand == "right"


@dataclass(frozen=True, slots=True)
class LayerAccess:
    """Describes how a layer is accessed from the base layer."""
    target_layer: int
    source_layer: int
    source_x: float
    source_y: float
    hand: str  # "left" or "right"
    is_momentary: bool  # True = hold to stay on layer, False = toggle/press
    access_key_label: str = ""


@dataclass(frozen=True, slots=True)
class Shortcut:
    """A shortcut/action that can be assigned to a position."""
    sid: int
    keys: str
    action: str
    app: str
    importance: float = 5.0
    category: str = "general"
    modifiers: Tuple[str, ...] = ()
    base_key: str = ""
    is_capability: bool = False
    is_l0_only: bool = False
    complexity: int = 1
    preferred_hand: str = "either"  # "left", "right", or "either"
    primary_app: Optional[str] = None
    app_demand: Dict[str, float] = field(default_factory=dict)

    @property
    def is_mod_tap(self) -> bool:
        return "&mt" in self.action or "mod_tap" in self.action

    @property
    def is_momentary(self) -> bool:
        return "&mo" in self.action or "momentary" in self.action


@dataclass(frozen=True, slots=True)
class UsageData:
    """Optional usage statistics from the AHK tracker."""
    sequences: Dict[str, dict] = field(default_factory=dict)
    chains: Dict[str, dict] = field(default_factory=dict)
    workflows: Dict[str, dict] = field(default_factory=dict)
    by_layer_shortcut: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_app: Dict[str, int] = field(default_factory=dict)
    app_time_seconds: Dict[str, float] = field(default_factory=dict)
    total_events: int = 0
    blind_spots: Dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Layout:
    """A complete layout assignment (immutable)."""
    genome: NDArray[np.int32]
    positions: Tuple[Position, ...]
    shortcuts: Tuple[Shortcut, ...]
    frozen_mask: NDArray[np.bool_]
    layer_to_indices: Dict[int, NDArray[np.int32]] = field(default_factory=dict)
    usage_data: UsageData = field(default_factory=UsageData)
    layer_access: Tuple[LayerAccess, ...] = field(default_factory=tuple)
    dynamic_groups: Tuple[Dict, ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        assert self.genome.shape[0] == len(self.positions), "Genome length mismatch"
        assert self.frozen_mask.shape[0] == len(self.positions), "Frozen mask length mismatch"
        
    @property
    def n_positions(self) -> int:
        return len(self.positions)
    
    @property
    def n_shortcuts(self) -> int:
        return len(self.shortcuts)
    
    @property
    def n_assigned(self) -> int:
        return int(np.sum(self.genome >= 0))
    
    @property
    def mutable_indices(self) -> NDArray[np.int32]:
        return np.where(~self.frozen_mask)[0].astype(np.int32)
    
    @property
    def frozen_indices(self) -> NDArray[np.int32]:
        return np.where(self.frozen_mask)[0].astype(np.int32)
    
    def get_assigned_at(self, idx: int) -> Optional[Shortcut]:
        sid = int(self.genome[idx])
        if sid < 0 or sid >= len(self.shortcuts):
            return None
        return self.shortcuts[sid]
    
    def get_position_of(self, sid: int) -> Optional[int]:
        idx = np.where(self.genome == sid)[0]
        return int(idx[0]) if len(idx) > 0 else None
    
    def get_positions_on_layer(self, layer: int) -> NDArray[np.int32]:
        if layer in self.layer_to_indices:
            return self.layer_to_indices[layer]
        indices = np.array([i for i, p in enumerate(self.positions) if p.layer == layer], dtype=np.int32)
        return indices
    
    def get_occupied_thumbs(self, layer: int, visited: set = None) -> List[str]:
        """Return list of occupied thumb hands for a given layer.
        
        Traces the access chain back to L0 to determine which thumbs are
        occupied by momentary layer access keys. For intermediate accesses,
        the hand of EACH momentary access button is included (not just the
        bottom-most one).
        """
        if visited is None:
            visited = set()
        if layer in visited:
            return []  # prevent cycles
        visited.add(layer)
        
        occupied = set()
        for access in self.layer_access:
            if access.target_layer != layer:
                continue
            if not access.is_momentary:
                continue
            # The hand holding THIS access button is occupied
            occupied.add(access.hand)
            # Also trace back to the source layer for nested accesses
            if access.source_layer != 0:
                source_occupied = self.get_occupied_thumbs(access.source_layer, visited.copy())
                occupied.update(source_occupied)
        return list(occupied)
    
    def get_full_access_occupancy(self, layer: int, visited: set = None) -> List[str]:
        """Return which hands are occupied by momentary access buttons along ANY path from L0.
        
        Unlike get_occupied_thumbs, this traces through ALL access types (toggle, lock, etc.)
        to determine if a right-hand momentary button must be held to reach this layer.
        
        Example: L9 is toggled from L4, but L4 requires a right-hand momentary from L0.
        So get_full_access_occupancy(9) returns ["right"].
        """
        if visited is None:
            visited = set()
        if layer == 0:
            return []
        if layer in visited:
            return []
        visited.add(layer)
        
        occupied = set()
        for access in self.layer_access:
            if access.target_layer != layer:
                continue
            # Trace back to source layer regardless of access type
            if access.source_layer != 0:
                source_occupied = self.get_full_access_occupancy(access.source_layer, visited.copy())
                occupied.update(source_occupied)
            # If this specific access is momentary, add its hand
            if access.is_momentary:
                occupied.add(access.hand)
        return list(occupied)
    
    def clone_with(self, genome: Optional[NDArray[np.int32]] = None) -> "Layout":
        return Layout(
            genome=genome.copy() if genome is not None else self.genome.copy(),
            positions=self.positions,
            shortcuts=self.shortcuts,
            frozen_mask=self.frozen_mask,
            layer_to_indices=self.layer_to_indices,
            usage_data=self.usage_data,
            layer_access=self.layer_access,
            dynamic_groups=self.dynamic_groups,
        )
    
    def is_valid(self) -> bool:
        assigned = self.genome[self.genome >= 0]
        return np.all(assigned < len(self.shortcuts))


@dataclass(frozen=True, slots=True)
class FitnessResult:
    """Result of a fitness evaluation."""
    objectives: NDArray[np.float32]
    factor_scores: Dict[str, float]
    total_score: float
    
    @property
    def effort(self) -> float:
        return float(self.objectives[0])
    
    @property
    def adjacency(self) -> float:
        return float(self.objectives[1])
    
    @property
    def violations(self) -> float:
        return float(self.objectives[2])
