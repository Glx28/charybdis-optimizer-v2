"""Semantic shortcut cluster detection for the Charybdis optimizer.

Clusters are discovered from the shortcut corpus by reading action *semantics*,
not only raw key names.  For example, "Snap window left" (Win+Left) and
"Snap window right" (Win+Right) form a left/right cluster even though the
actual key combinations are not simple arrow keys.

Detected clusters become strong soft-pressure groups in the fitness kernel and
critical small clusters (Copy/Paste, Undo/Redo, raw directional pairs) are also
candidates for atomic group-move mutations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core import Shortcut


# ---------------------------------------------------------------------------
# Directional / sequence vocabulary
# ---------------------------------------------------------------------------

# Each token maps to a canonical order and a relative offset from the anchor
# (order 0).  Horizontal pairs use (dx, dy); vertical pairs use (0, 1).
DIRECTION_TOKENS = {
    # horizontal: earlier / left
    "left": (0, 0.0, 0.0),
    "previous": (0, 0.0, 0.0),
    "prev": (0, 0.0, 0.0),
    "back": (0, 0.0, 0.0),
    "backward": (0, 0.0, 0.0),
    "before": (0, 0.0, 0.0),
    "earlier": (0, 0.0, 0.0),
    # horizontal: later / right
    "right": (1, 1.0, 0.0),
    "next": (1, 1.0, 0.0),
    "forward": (1, 1.0, 0.0),
    "forwards": (1, 1.0, 0.0),
    "after": (1, 1.0, 0.0),
    "later": (1, 1.0, 0.0),
    # vertical: earlier / up
    "up": (0, 0.0, 0.0),
    "upward": (0, 0.0, 0.0),
    # vertical: later / down
    "down": (1, 0.0, 1.0),
    "downward": (1, 0.0, 1.0),
    # zoom / scale
    "in": (0, 0.0, 0.0),
    "out": (1, 1.0, 0.0),
}

# Sequence tokens where the word itself implies order (not spatial direction).
# Value: (order, dx, dy, family).  Family groups related sequence words so that
# "Cut" + "Copy" + "Paste" form one cluster and "Undo" + "Redo" form another.
SEQUENCE_TOKENS = {
    "cut": (-1, -1.0, 0.0, "clipboard"),
    "copy": (0, 0.0, 0.0, "clipboard"),
    "paste": (1, 1.0, 0.0, "clipboard"),
    "undo": (0, 0.0, 0.0, "history"),
    "redo": (1, 1.0, 0.0, "history"),
}

# Raw key names that form obvious directional / sequential pairs.
RAW_KEY_CLUSTERS: Dict[str, List[Tuple[str, int, float, float]]] = {
    "page": [
        ("PageUp", 0, 0.0, 0.0),
        ("PageDown", 1, 0.0, 1.0),
    ],
    "home_end": [
        ("Home", 0, 0.0, 0.0),
        ("End", 1, 1.0, 0.0),
    ],
    "volume": [
        ("VolumeUp", 0, 0.0, 0.0),
        ("VolumeDown", 1, 0.0, 1.0),
    ],
}

# Shortcuts we never cluster (noise/scroll fakes).
IGNORED_ACTION_WORDS = {"scroll up", "scroll down", "scrollup", "scrolldown"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterMember:
    sid: int
    order: int
    dx: float
    dy: float


@dataclass
class SemanticCluster:
    name: str
    category: str
    members: List[ClusterMember] = field(default_factory=list)
    weight: float = 1.0
    is_critical: bool = False

    @property
    def member_sids(self) -> List[int]:
        return [m.sid for m in self.members]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]+")


def _normalize_action(action: str) -> str:
    """Lowercase and strip punctuation from an action string."""
    return _RE_NON_ALPHANUM.sub(" ", (action or "").lower()).strip()


def _tokenize(action: str) -> List[str]:
    return _normalize_action(action).split()


def _extract_direction(action: str) -> Optional[Tuple[str, int, float, float, str]]:
    """Return (token, order, dx, dy, family) if the action contains a directional word."""
    tokens = _tokenize(action)
    for tok in tokens:
        if tok in DIRECTION_TOKENS:
            info = DIRECTION_TOKENS[tok]
            return (tok, info[0], info[1], info[2], "direction")
    return None


def _extract_sequence(action: str) -> Optional[Tuple[str, int, float, float, str]]:
    """Return (token, order, dx, dy, family) if the action contains a sequence word."""
    tokens = _tokenize(action)
    for tok in tokens:
        if tok in SEQUENCE_TOKENS:
            return (tok,) + SEQUENCE_TOKENS[tok]
    return None


def _action_stem(action: str, token: str) -> str:
    """Return the action with the directional/sequence token removed."""
    norm = _normalize_action(action)
    # Remove the token as a whole word.
    pattern = r"\b" + re.escape(token) + r"\b"
    stem = re.sub(pattern, "", norm)
    # Collapse whitespace.
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _make_cluster_name(stem: str, category: str, family: str = "", tokens: List[str] = None) -> str:
    clean = re.sub(r"\s+", "_", stem).strip("_")
    if not clean:
        # Sequence clusters get their semantic family name (clipboard, history)
        # for clarity; direction clusters fall back to the directional tokens.
        if category == "sequence" and family:
            clean = family
        elif tokens:
            clean = "_".join(sorted(set(tokens)))
        else:
            clean = family or category
    return f"{category}_{clean}"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_semantic_clusters(shortcuts: List[Shortcut]) -> List[SemanticCluster]:
    """Build semantic clusters from the shortcut corpus.

    Returns a list of clusters.  Each cluster contains member SIDs, a canonical
    order, and canonical relative offsets.  Critical clusters are flagged for
    possible atomic group-move treatment.
    """
    by_sid = {s.sid: s for s in shortcuts}

    # Raw arrow keys already have their own protected cluster; skip them here
    # so we do not fight the existing arrow-shape machinery.
    RAW_ARROW_KEYS = {"LeftArrow", "RightArrow", "UpArrow", "DownArrow"}

    def _skip_shortcut(sc: Shortcut) -> bool:
        if sc.is_l0_only or sc.is_layer_access or not sc.action:
            return True
        if sc.category == "mouse":
            return True
        if not sc.modifiers and (sc.base_key or "").strip() in RAW_ARROW_KEYS:
            return True
        return False

    # Action-stem based clusters (directional pairs/groups).
    # Key: (stem, family).  Family is "direction" for all direction words.
    stem_groups: Dict[Tuple[str, str], List[Tuple[int, str, int, float, float]]] = {}
    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        norm = _normalize_action(sc.action)
        if any(ig in norm for ig in IGNORED_ACTION_WORDS):
            continue
        dir_info = _extract_direction(sc.action)
        if dir_info is None:
            continue
        token, order, dx, dy, family = dir_info
        stem = _action_stem(sc.action, token)
        if not stem:
            stem = "_"
        key = (stem, family)
        stem_groups.setdefault(key, []).append((sc.sid, token, order, dx, dy))

    # Action-stem based sequence clusters (copy/paste, undo/redo).
    # Key: (stem, family).  Family comes from SEQUENCE_TOKENS so that "Cut",
    # "Copy", "Paste" cluster together and "Undo", "Redo" cluster together.
    seq_groups: Dict[Tuple[str, str], List[Tuple[int, str, int, float, float]]] = {}
    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        seq_info = _extract_sequence(sc.action)
        if seq_info is None:
            continue
        token, order, dx, dy, family = seq_info
        stem = _action_stem(sc.action, token)
        if not stem:
            stem = "_"
        key = (stem, family)
        seq_groups.setdefault(key, []).append((sc.sid, token, order, dx, dy))

    clusters: List[SemanticCluster] = []

    def _build_cluster(key, entries, category):
        # Within one stem/family, keep only the highest-importance shortcut for
        # each distinct order.  This prevents e.g. two "Find previous" bindings
        # from both claiming the same canonical anchor position.
        best_by_order: Dict[int, Tuple[int, str, int, float, float]] = {}
        for entry in entries:
            sid, token, order, dx, dy = entry
            existing = best_by_order.get(order)
            if existing is None or by_sid[sid].importance > by_sid[existing[0]].importance:
                best_by_order[order] = entry

        if len(best_by_order) < 2:
            return None

        # Pick the lowest order as anchor; rebase offsets so anchor is at (0,0).
        min_order = min(best_by_order.keys())
        anchor_dx = anchor_dy = None
        for e in best_by_order.values():
            if e[2] == min_order:
                anchor_dx, anchor_dy = e[3], e[4]
                break
        if anchor_dx is None:
            return None

        members = []
        for entry in sorted(best_by_order.values(), key=lambda x: x[2]):
            sid, token, order, dx, dy = entry
            members.append(ClusterMember(
                sid=sid,
                order=order,
                dx=dx - anchor_dx,
                dy=dy - anchor_dy,
            ))
        stem, family = key
        tokens = [e[1] for e in best_by_order.values()]
        # For sequence clusters, prefer the semantic family name (clipboard, history)
        # when the action stem is empty.
        default_name = family if category == "sequence" else ""
        name = _make_cluster_name(stem, category, default_name, tokens)
        total_importance = sum(by_sid[m.sid].importance for m in members)
        return SemanticCluster(
            name=name,
            category=category,
            members=members,
            weight=min(10.0, 1.0 + total_importance * 0.2),
            is_critical=(category == "sequence" and len(members) >= 2),
        )

    for key, entries in stem_groups.items():
        cluster = _build_cluster(key, entries, "direction")
        if cluster is not None:
            clusters.append(cluster)

    for key, entries in seq_groups.items():
        cluster = _build_cluster(key, entries, "sequence")
        if cluster is not None:
            clusters.append(cluster)

    # Raw key clusters (PageUp/PageDown, Home/End, VolumeUp/VolumeDown).
    base_to_sid: Dict[str, int] = {}
    for sc in shortcuts:
        if sc.is_l0_only or sc.is_layer_access or sc.modifiers:
            continue
        base = (sc.base_key or "").strip()
        if base:
            base_to_sid[base] = sc.sid

    for category, members in RAW_KEY_CLUSTERS.items():
        cluster_members = []
        for base, order, dx, dy in members:
            sid = base_to_sid.get(base)
            if sid is None:
                continue
            cluster_members.append((sid, order, dx, dy))
        if len(cluster_members) < 2:
            continue
        # Rebase to anchor.
        min_order = min(e[1] for e in cluster_members)
        anchor_dx = anchor_dy = None
        for e in cluster_members:
            if e[1] == min_order:
                anchor_dx, anchor_dy = e[2], e[3]
                break
        members_out = [
            ClusterMember(sid=sid, order=order, dx=dx - anchor_dx, dy=dy - anchor_dy)
            for sid, order, dx, dy in cluster_members
        ]
        total_importance = sum(by_sid[m.sid].importance for m in members_out)
        clusters.append(SemanticCluster(
            name=f"raw_{category}",
            category="raw_key",
            members=members_out,
            weight=min(10.0, 1.0 + total_importance * 0.2),
            is_critical=False,
        ))

    # Deduplicate: if the same SID appears in multiple clusters, keep the more
    # specific cluster (lower member count wins, then higher weight).
    sid_to_best_cluster: Dict[int, SemanticCluster] = {}
    kept: List[SemanticCluster] = []
    for cluster in clusters:
        conflict = False
        for m in cluster.members:
            existing = sid_to_best_cluster.get(m.sid)
            if existing is not None:
                existing_score = (len(existing.members), -existing.weight)
                new_score = (len(cluster.members), -cluster.weight)
                if new_score < existing_score:
                    conflict = True
                    break
        if conflict:
            continue
        for m in cluster.members:
            sid_to_best_cluster[m.sid] = cluster
        kept.append(cluster)

    # Remove clusters that lost members during deduplication.
    final: List[SemanticCluster] = []
    for cluster in kept:
        if all(sid_to_best_cluster.get(m.sid) is cluster for m in cluster.members):
            final.append(cluster)

    return final
