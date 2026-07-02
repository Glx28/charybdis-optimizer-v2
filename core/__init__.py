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
    """Legacy static snapshot of a layer-access position.

    NOT authoritative for evolved layouts.  The optimizer treats layer-access
    buttons as first-class genome capabilities: any shortcut with
    ``is_layer_access=True`` can be placed anywhere in the mutable genome.
    Scoring and acceptance always infer access from the live genome, never from
    this static record.  New layouts should leave ``Layout.layer_access`` empty.
    """
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
    is_layer_access: bool = False
    access_target_layer: int = -1
    access_is_momentary: bool = False

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
    shortcut_sequences: Dict[str, dict] = field(default_factory=dict)
    shortcut_workflows: Dict[str, dict] = field(default_factory=dict)
    app_sequences: Dict[str, dict] = field(default_factory=dict)
    app_workflows: Dict[str, dict] = field(default_factory=dict)
    shortcuts: Dict[str, dict] = field(default_factory=dict)
    mouse_session_shortcuts: Dict[str, dict] = field(default_factory=dict)
    mouse_clicks: Dict[str, dict] = field(default_factory=dict)
    scroll_total: int = 0
    scroll_by_layer: Dict[str, int] = field(default_factory=dict)
    raw_completion_keys: Dict[str, dict] = field(default_factory=dict)
    raw_completion_total: int = 0
    by_layer_shortcut: Dict[str, Dict[str, int]] = field(default_factory=dict)
    layer_shortcuts: Dict[str, dict] = field(default_factory=dict)
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
    # Legacy static access list — normally empty and NOT authoritative.
    # Scoring and acceptance use genome shortcuts (shortcut.is_layer_access) instead.
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
    
    def get_occupied_thumbs_from_genome(self, layer: int, visited: set = None) -> List[str]:
        """Infer occupied thumb hands from the evolved genome bindings.

        Iterates over assigned shortcuts with ``is_layer_access=True`` to find
        which thumb buttons must be held to reach ``layer``.  This is the
        authoritative path for evolved candidates; it always reflects the
        current genome, not any static access list.
        """
        if visited is None:
            visited = set()
        if layer in visited:
            return []
        visited.add(layer)

        occupied: set = set()
        for idx, sid in enumerate(self.genome):
            sid = int(sid)
            if sid < 0 or sid >= len(self.shortcuts):
                continue
            shortcut = self.shortcuts[sid]
            if not shortcut.is_layer_access or shortcut.access_target_layer != layer:
                continue
            if not shortcut.access_is_momentary:
                continue
            pos = self.positions[idx]
            occupied.add(pos.hand)
            source_layer = pos.layer
            if source_layer != 0:
                occupied.update(self.get_occupied_thumbs_from_genome(source_layer, visited.copy()))
        return list(occupied)

    def get_full_access_occupancy_from_genome(self, layer: int, visited: set = None) -> List[str]:
        """Infer hand occupancy for ALL access types from evolved genome bindings.

        Traces through toggle and momentary access to determine if any
        right-hand momentary button must be held along any path to ``layer``.
        Authoritative for evolved candidates; reads the genome, not the
        static access list.
        """
        if visited is None:
            visited = set()
        if layer == 0:
            return []
        if layer in visited:
            return []
        visited.add(layer)

        occupied: set = set()
        for idx, sid in enumerate(self.genome):
            sid = int(sid)
            if sid < 0 or sid >= len(self.shortcuts):
                continue
            shortcut = self.shortcuts[sid]
            if not shortcut.is_layer_access or shortcut.access_target_layer != layer:
                continue
            pos = self.positions[idx]
            source_layer = pos.layer
            if source_layer != 0:
                occupied.update(self.get_full_access_occupancy_from_genome(source_layer, visited.copy()))
            if shortcut.access_is_momentary:
                occupied.add(pos.hand)
        return list(occupied)

    def get_occupied_thumbs(self, layer: int, visited: set = None) -> List[str]:
        """Legacy/static fallback only. Not authoritative for evolved dynamic access.

        Reads ``self.layer_access`` which is a legacy static fallback.  An
        evolved candidate may have placed access shortcuts in completely
        different positions or to different layers.  Call
        ``get_occupied_thumbs_from_genome`` instead for scoring/reporting.
        """
        if visited is None:
            visited = set()
        if layer in visited:
            return []
        visited.add(layer)

        occupied = set()
        for access in self.layer_access:
            if access.target_layer != layer:
                continue
            if not access.is_momentary:
                continue
            occupied.add(access.hand)
            if access.source_layer != 0:
                source_occupied = self.get_occupied_thumbs(access.source_layer, visited.copy())
                occupied.update(source_occupied)
        return list(occupied)

    def get_full_access_occupancy(self, layer: int, visited: set = None) -> List[str]:
        """Legacy/static fallback only. Not authoritative for evolved dynamic access.

        Reads ``self.layer_access`` which is a legacy static fallback.  Call
        ``get_full_access_occupancy_from_genome`` instead for scoring/reporting
        of evolved candidates.
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
            if access.source_layer != 0:
                source_occupied = self.get_full_access_occupancy(access.source_layer, visited.copy())
                occupied.update(source_occupied)
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
    constraints: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    
    @property
    def effort(self) -> float:
        return float(self.objectives[0])
    
    @property
    def adjacency(self) -> float:
        return float(self.objectives[1])
    
    @property
    def violations(self) -> float:
        return float(self.objectives[2])
