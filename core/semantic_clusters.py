"""Semantic shortcut cluster detection for the Charybdis optimizer.

Clusters are discovered from the shortcut corpus by reading action *semantics*,
raw key patterns, and cross-app equivalents — not only raw key names.  For
example, "Snap window left" (Win+Left) and "Snap window right" (Win+Right)
form a left/right cluster even though the actual key combinations are not
simple arrow keys.

Detected clusters become strong soft-pressure groups in the fitness kernel and
critical small clusters (Copy/Paste, Undo/Redo, raw directional pairs, etc.)
are candidates for atomic group-move mutations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from core import Shortcut


# ---------------------------------------------------------------------------
# Directional / sequence / antonym vocabulary
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
    "first": (0, 0.0, 0.0),
    "home": (0, 0.0, 0.0),
    "start": (0, 0.0, 0.0),
    "beginning": (0, 0.0, 0.0),
    # horizontal: later / right
    "right": (1, 1.0, 0.0),
    "next": (1, 1.0, 0.0),
    "forward": (1, 1.0, 0.0),
    "forwards": (1, 1.0, 0.0),
    "after": (1, 1.0, 0.0),
    "later": (1, 1.0, 0.0),
    "last": (1, 1.0, 0.0),
    "end": (1, 1.0, 0.0),
    # vertical: earlier / up
    "up": (0, 0.0, 0.0),
    "upward": (0, 0.0, 0.0),
    "above": (0, 0.0, 0.0),
    "top": (0, 0.0, 0.0),
    # vertical: later / down
    "down": (1, 0.0, 1.0),
    "downward": (1, 0.0, 1.0),
    "below": (1, 0.0, 1.0),
    "bottom": (1, 0.0, 1.0),
    # zoom / scale
    "in": (0, 0.0, 0.0),
    "out": (1, 1.0, 0.0),
    # open/close, show/hide, enable/disable
    "open": (0, 0.0, 0.0),
    "show": (0, 0.0, 0.0),
    "enable": (0, 0.0, 0.0),
    "close": (1, 1.0, 0.0),
    "hide": (1, 1.0, 0.0),
    "disable": (1, 1.0, 0.0),
    # size / state
    "maximize": (1, 1.0, 0.0),
    "minimize": (0, 0.0, 0.0),
    "restore": (0, 0.0, 0.0),
    "shrink": (0, 0.0, 0.0),
    "expand": (1, 1.0, 0.0),
    "increase": (1, 1.0, 0.0),
    "decrease": (0, 0.0, 0.0),
    "raise": (0, 0.0, 0.0),
    "lower": (1, 0.0, 1.0),
    "indent": (1, 1.0, 0.0),
    "outdent": (0, 0.0, 0.0),
    "add": (1, 1.0, 0.0),
    "remove": (0, 0.0, 0.0),
    "delete": (0, 0.0, 0.0),
    "insert": (1, 1.0, 0.0),
    "upper": (1, 1.0, 0.0),
    "lower": (0, 0.0, 0.0),
    "start": (0, 0.0, 0.0),
    "stop": (1, 1.0, 0.0),
    "play": (0, 0.0, 0.0),
    "pause": (1, 1.0, 0.0),
    "mute": (0, 0.0, 0.0),
    "unmute": (1, 1.0, 0.0),
    "lock": (0, 0.0, 0.0),
    "unlock": (1, 1.0, 0.0),
}

# Sequence tokens where the word itself implies order (not spatial direction).
# Value: (order, dx, dy, family).  Family groups related sequence words so that
# "Cut", "Copy", "Paste" form one cluster and "Undo", "Redo" form another.
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
    "arrow_horizontal": [
        ("LeftArrow", 0, 0.0, 0.0),
        ("RightArrow", 1, 1.0, 0.0),
    ],
    "arrow_vertical": [
        ("UpArrow", 0, 0.0, 0.0),
        ("DownArrow", 1, 0.0, 1.0),
    ],
}

# Shortcuts we never cluster (noise/scroll fakes).
IGNORED_ACTION_WORDS = {"scroll up", "scroll down", "scrollup", "scrolldown"}


# ---------------------------------------------------------------------------
# Semantic keyword families
# ---------------------------------------------------------------------------
#
# A family matches shortcuts whose actions contain any of the listed keywords.
# If ``direction_tokens`` is provided, members are ordered spatially using those
# tokens; otherwise all matched shortcuts simply form a compactness cluster.

@dataclass(frozen=True)
class KeywordFamily:
    name: str
    keywords: Tuple[str, ...]
    direction_tokens: Optional[Tuple[str, ...]] = None
    require_app: Optional[str] = None
    min_members: int = 2
    exclude_keywords: Tuple[str, ...] = ()


KEYWORD_FAMILIES = [
    # Clipboard (already partly covered by SEQUENCE_TOKENS, but this catches
    # app-specific phrasing like "Paste without formatting" and "Clipboard history").
    # Exclude line-copying actions which are editor-specific, not clipboard.
    KeywordFamily("clipboard", ("cut", "copy", "paste", "clipboard"), exclude_keywords=("line",), min_members=2),
    # History (Undo/Redo).
    KeywordFamily("history", ("undo", "redo")),
    # Browser tabs.
    KeywordFamily("browser_tab", ("tab",), direction_tokens=("new", "close", "reopen", "next", "previous", "last", "switch")),
    # Browser navigation.
    KeywordFamily("browser_nav", ("back", "forward")),
    # Browser refresh/reload.
    KeywordFamily("browser_refresh", ("refresh", "reload", "hard refresh")),
    # Browser find.
    KeywordFamily("browser_find", ("find",), direction_tokens=("previous", "next", "on page")),
    # Browser zoom.
    KeywordFamily("browser_zoom", ("zoom",), direction_tokens=("in", "out", "reset")),
    # Browser DevTools.
    KeywordFamily("browser_devtools", ("devtools", "console", "inspect element", "inspect")),
    # Browser bookmarks.
    KeywordFamily("browser_bookmark", ("bookmark", "bookmarks", "favorites")),
    # Browser address / search bar.
    KeywordFamily("browser_address", ("address bar", "focus address", "addressbar")),
    # Browser window management.
    KeywordFamily("browser_window", ("new window", "incognito", "inprivate")),
    # Browser file operations.
    KeywordFamily("browser_file_ops", ("print", "save page as")),
    # Browser panels / sidebar.
    KeywordFamily("browser_panel", ("history", "downloads", "sidebar")),
    # Browser fullscreen / view toggles.
    KeywordFamily("browser_view", ("fullscreen", "device toolbar")),
    # Window management (Windows).
    KeywordFamily("window_state", ("maximize", "minimize", "restore", "close window", "snap window")),
    # Virtual desktops.
    KeywordFamily("virtual_desktop", ("virtual desktop", "desktop", "switch desktop")),
    # Monitor movement.
    KeywordFamily("monitor_move", ("monitor",), direction_tokens=("left", "right")),
    # Launchers / search.
    KeywordFamily("launcher", ("search", "run dialog", "launcher", "command palette")),
    # PowerToys tools (all in the PowerToys app).
    KeywordFamily("powertoys", (), require_app="PowerToys"),
    # Text formatting.
    KeywordFamily("text_format", ("bold", "italic", "underline", "strikethrough")),
    # Font size.
    KeywordFamily("font_size", ("font size",), direction_tokens=("increase", "decrease")),
    # Excel navigation edges.
    KeywordFamily("excel_edge", ("edge of data", "select to", "jump to"), direction_tokens=("left", "right", "top", "bottom"), exclude_keywords=("bracket",)),
    # Excel cell formatting.
    KeywordFamily("excel_format", ("format",), require_app="Microsoft Excel"),
    # VS Code cursor vertical.
    KeywordFamily("vscode_cursor", ("add cursor",), direction_tokens=("above", "below", "up", "down")),
    # VS Code selection expand/shrink.
    KeywordFamily("vscode_selection", ("expand selection", "shrink selection")),
    # VS Code indent/outdent.
    KeywordFamily("vscode_indent", ("indent line", "outdent line")),
    # VS Code copy line vertical.
    KeywordFamily("vscode_copy_line", ("copy line",), direction_tokens=("up", "down")),
    # VS Code debug step.
    KeywordFamily("vscode_debug", ("step over", "step out", "step into")),
    # VS Code editor tools (peek, split, comment, terminal, bracket jump).
    KeywordFamily("vscode_editor", ("peek definition", "split editor", "toggle block comment", "new terminal", "jump to matching bracket")),
    # VS Code terminal.
    KeywordFamily("vscode_terminal", ("terminal",)),
    # Terminal split panes.
    KeywordFamily("terminal_split", ("split pane",)),
    # Presentation slides.
    KeywordFamily("presentation", ("slide",), direction_tokens=("previous", "next", "current")),
    # Teams sections.
    KeywordFamily("teams_section", ("section",), direction_tokens=("previous", "next")),
    # File Explorer view modes.
    KeywordFamily("explorer_view", ("icons", "large icons", "extra large icons")),
    # File Explorer panes.
    KeywordFamily("explorer_pane", ("pane", "preview pane", "details pane"), require_app="File Explorer"),
    # Emoji picker.
    KeywordFamily("emoji", ("emoji picker", "emoji")),
    # Screenshots / snips.
    KeywordFamily("screenshot", ("screenshot", "snip")),
    # Windows system tools (clipboard history, screenshot, task manager).
    KeywordFamily("windows_tools", ("clipboard history", "screenshot", "task manager")),
    # System settings / info.
    KeywordFamily("system_settings", ("settings", "system info")),
    # Input language.
    KeywordFamily("input_language", ("input language", "keyboard layout")),
    # Teams call controls.
    KeywordFamily("teams_call", ("mute", "deafen", "video", "raise hand", "lower hand")),
    # M-Files workflow.
    KeywordFamily("mfiles", (), require_app="M-Files Desktop Client"),
    # Raw completion keys.
    KeywordFamily("raw_completion", (), require_app="Raw Keys"),
    # Mouse arrows.
    KeywordFamily("mouse_arrows", (), require_app="Mouse"),
]


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
    pattern = r"\b" + re.escape(token) + r"\b"
    stem = re.sub(pattern, "", norm)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _make_cluster_name(stem: str, category: str, family: str = "", tokens: List[str] = None) -> str:
    clean = re.sub(r"\s+", "_", stem).strip("_")
    if not clean:
        if category == "sequence" and family:
            clean = family
        elif tokens:
            clean = "_".join(sorted(set(tokens)))
        else:
            clean = family or category
    return f"{category}_{clean}"


# ---------------------------------------------------------------------------
# Detection strategies
# ---------------------------------------------------------------------------

def _skip_shortcut(sc: Shortcut) -> bool:
    if sc.is_l0_only or sc.is_layer_access or not sc.action:
        return True
    if sc.category == "mouse":
        return True
    return False


def _build_cluster(
    name: str,
    category: str,
    by_sid: Dict[int, Shortcut],
    entries: Iterable[Tuple[int, int, float, float]],
    is_critical: bool = False,
    compactness: bool = False,
) -> Optional[SemanticCluster]:
    """Build a cluster from (sid, order, dx, dy) entries.

    For directional clusters, keeps only the highest-importance shortcut for
    each distinct order so that e.g. two "Find previous" bindings do not both
    claim the same anchor.  For compactness-only clusters, all entries are kept
    because the goal is simply to place related shortcuts on the same layer.
    """
    if compactness:
        # Keep all entries; collapse offsets to (0,0) since relative position
        # within the cluster is undefined.
        kept = [(sid, 0, 0.0, 0.0) for sid, _order, _dx, _dy in entries]
    else:
        best_by_order: Dict[int, Tuple[int, int, float, float]] = {}
        for sid, order, dx, dy in entries:
            existing = best_by_order.get(order)
            if existing is None or by_sid[sid].importance > by_sid[existing[0]].importance:
                best_by_order[order] = (sid, order, dx, dy)
        kept = list(best_by_order.values())

    if len(kept) < 2:
        return None

    min_order = min(e[1] for e in kept)
    anchor_dx = anchor_dy = None
    for e in kept:
        if e[1] == min_order:
            anchor_dx, anchor_dy = e[2], e[3]
            break
    if anchor_dx is None:
        return None

    members = []
    for entry in sorted(kept, key=lambda x: x[1]):
        sid, order, dx, dy = entry
        members.append(ClusterMember(
            sid=sid,
            order=order,
            dx=dx - anchor_dx,
            dy=dy - anchor_dy,
        ))
    total_importance = sum(by_sid[m.sid].importance for m in members)
    return SemanticCluster(
        name=name,
        category=category,
        members=members,
        weight=min(15.0, 1.0 + total_importance * 0.25),
        is_critical=is_critical,
    )


def _direction_and_sequence_clusters(
    shortcuts: List[Shortcut], by_sid: Dict[int, Shortcut]
) -> List[SemanticCluster]:
    """Action-stem clusters using DIRECTION_TOKENS and SEQUENCE_TOKENS."""
    stem_groups: Dict[Tuple[str, str], List[Tuple[int, str, int, float, float]]] = {}
    seq_groups: Dict[Tuple[str, str], List[Tuple[int, str, int, float, float]]] = {}

    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        norm = _normalize_action(sc.action)
        if any(ig in norm for ig in IGNORED_ACTION_WORDS):
            continue

        dir_info = _extract_direction(sc.action)
        if dir_info is not None:
            token, order, dx, dy, family = dir_info
            stem = _action_stem(sc.action, token) or "_"
            stem_groups.setdefault((stem, family), []).append((sc.sid, token, order, dx, dy))

        seq_info = _extract_sequence(sc.action)
        if seq_info is not None:
            token, order, dx, dy, family = seq_info
            stem = _action_stem(sc.action, token) or "_"
            seq_groups.setdefault((stem, family), []).append((sc.sid, token, order, dx, dy))

    clusters: List[SemanticCluster] = []

    def _from_entries(key, entries, category):
        # Convert (sid, token, order, dx, dy) -> (sid, order, dx, dy) using token-specific offsets.
        best_by_order: Dict[int, Tuple[int, str, int, float, float]] = {}
        for sid, token, order, dx, dy in entries:
            existing = best_by_order.get(order)
            if existing is None or by_sid[sid].importance > by_sid[existing[0]].importance:
                best_by_order[order] = (sid, token, order, dx, dy)
        if len(best_by_order) < 2:
            return None
        stem, family = key
        tokens = [e[1] for e in best_by_order.values()]
        default_name = family if category == "sequence" else ""
        name = _make_cluster_name(stem, category, default_name, tokens)
        build_entries = [(e[0], e[2], e[3], e[4]) for e in best_by_order.values()]
        return _build_cluster(name, category, by_sid, build_entries, is_critical=(category == "sequence"))

    for key, entries in stem_groups.items():
        cluster = _from_entries(key, entries, "direction")
        if cluster is not None:
            clusters.append(cluster)

    for key, entries in seq_groups.items():
        cluster = _from_entries(key, entries, "sequence")
        if cluster is not None:
            clusters.append(cluster)

    return clusters


def _keyword_family_clusters(
    shortcuts: List[Shortcut], by_sid: Dict[int, Shortcut]
) -> List[SemanticCluster]:
    """Clusters from semantic keyword families."""
    clusters: List[SemanticCluster] = []

    # Pre-compile whole-word patterns for each family.
    family_patterns = {
        family: [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in family.keywords]
        for family in KEYWORD_FAMILIES
    }
    family_excludes = {
        family: [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in family.exclude_keywords]
        for family in KEYWORD_FAMILIES
    }

    for family in KEYWORD_FAMILIES:
        matched: List[Shortcut] = []
        patterns = family_patterns[family]
        for sc in shortcuts:
            if _skip_shortcut(sc):
                continue
            if family.require_app is not None and sc.app != family.require_app:
                continue
            norm = _normalize_action(sc.action)
            if family.require_app is not None and not family.keywords:
                # App-only family: include all non-layer-access shortcuts from that app.
                matched.append(sc)
                continue
            if any(p.search(norm) for p in patterns):
                if any(p.search(norm) for p in family_excludes[family]):
                    continue
                matched.append(sc)

        if len(matched) < family.min_members:
            continue

        # Build the family as a compactness cluster.  Directional families
        # (e.g. browser_tab) would ideally carry relative offsets, but the
        # primary goal is to keep the whole semantic family on one layer; the
        # separate directional-stem detector still creates ordered sub-clusters
        # for strongly left/right pairs.
        entries = [(sc.sid, 0, 0.0, 0.0) for sc in matched]
        cluster = _build_cluster(f"family_{family.name}", "keyword_family", by_sid, entries, compactness=True)
        if cluster is not None:
            clusters.append(cluster)

    return clusters


def _cross_app_exact_clusters(
    shortcuts: List[Shortcut], by_sid: Dict[int, Shortcut]
) -> List[SemanticCluster]:
    """Cluster shortcuts whose normalized action text is identical across apps.

    This catches e.g. "Settings", "Search", "Copy", "Paste", "Undo", "Redo"
    even when the corpus kept only one app instance per action.
    """
    by_action: Dict[str, List[Shortcut]] = {}
    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        norm = _normalize_action(sc.action)
        if len(norm) < 2:
            continue
        by_action.setdefault(norm, []).append(sc)

    clusters: List[SemanticCluster] = []
    for action, items in by_action.items():
        if len(items) < 2:
            continue
        total_importance = sum(s.importance for s in items)
        if total_importance < 1.0:
            continue
        entries = [(sc.sid, 0, 0.0, 0.0) for sc in items]
        cluster = _build_cluster(f"cross_app_{action.replace(' ', '_')}", "cross_app", by_sid, entries)
        if cluster is not None:
            clusters.append(cluster)

    return clusters


def _raw_key_clusters(
    shortcuts: List[Shortcut], by_sid: Dict[int, Shortcut]
) -> List[SemanticCluster]:
    """Raw key clusters (PageUp/PageDown, Home/End, VolumeUp/VolumeDown, arrows)."""
    base_to_sid: Dict[str, int] = {}
    for sc in shortcuts:
        if sc.is_l0_only or sc.is_layer_access or sc.modifiers:
            continue
        base = (sc.base_key or "").strip()
        if base:
            base_to_sid[base] = sc.sid

    clusters: List[SemanticCluster] = []
    for category, members in RAW_KEY_CLUSTERS.items():
        entries = []
        for base, order, dx, dy in members:
            sid = base_to_sid.get(base)
            if sid is not None:
                entries.append((sid, order, dx, dy))
        cluster = _build_cluster(f"raw_{category}", "raw_key", by_sid, entries)
        if cluster is not None:
            clusters.append(cluster)

    return clusters


def _key_pattern_clusters(
    shortcuts: List[Shortcut], by_sid: Dict[int, Shortcut]
) -> List[SemanticCluster]:
    """Clusters from repeated key patterns (numeric sequences, modifier families)."""
    clusters: List[SemanticCluster] = []

    # Modifier + number sequences (e.g. Ctrl+1..Ctrl+8, Win+1..Win+5).
    numeric_pattern = re.compile(r"^(Ctrl|Alt|Shift|Win)\+(\d+)$")
    by_mod: Dict[str, List[Tuple[int, int, Shortcut]]] = {}
    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        m = numeric_pattern.match(sc.keys)
        if m:
            mod, num = m.group(1), int(m.group(2))
            by_mod.setdefault(mod, []).append((num, sc.sid, sc))

    for mod, items in by_mod.items():
        if len(items) < 2:
            continue
        # Keep consecutive numeric runs of length >= 2.
        items_sorted = sorted(items, key=lambda x: x[0])
        run: List[Tuple[int, int, Shortcut]] = []
        for it in items_sorted:
            if not run or it[0] == run[-1][0] + 1:
                run.append(it)
            else:
                if len(run) >= 2:
                    entries = [(sid, num, float(num), 0.0) for num, sid, _ in run]
                    cluster = _build_cluster(f"pattern_{mod}_digits", "key_pattern", by_sid, entries)
                    if cluster is not None:
                        clusters.append(cluster)
                run = [it]
        if len(run) >= 2:
            entries = [(sid, num, float(num), 0.0) for num, sid, _ in run]
            cluster = _build_cluster(f"pattern_{mod}_digits", "key_pattern", by_sid, entries)
            if cluster is not None:
                clusters.append(cluster)

    return clusters


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _cluster_priority(cluster: SemanticCluster) -> float:
    """Return a priority score used to resolve SID conflicts.

    Explicit keyword families and critical sequence clusters win over auto-
    detected directional stem clusters, because the families encode the actual
    semantic group the user cares about.
    """
    category_bonus = {
        "sequence": 80.0 if cluster.is_critical else 40.0,
        "keyword_family": 60.0,
        "cross_app": 45.0,
        "direction": 25.0,
        "raw_key": 15.0,
        "key_pattern": 10.0,
    }.get(cluster.category, 5.0)
    return category_bonus + cluster.weight


def _deduplicate_clusters(clusters: List[SemanticCluster]) -> List[SemanticCluster]:
    """Keep the highest-priority cluster for each SID.

    If a cluster loses members during assignment and drops below two surviving
    members, it is discarded.
    """
    clusters_sorted = sorted(clusters, key=_cluster_priority, reverse=True)

    sid_to_best: Dict[int, SemanticCluster] = {}
    for cluster in clusters_sorted:
        for m in cluster.members:
            existing = sid_to_best.get(m.sid)
            if existing is None or _cluster_priority(cluster) > _cluster_priority(existing):
                sid_to_best[m.sid] = cluster

    # Rebuild clusters from surviving SIDs.
    cluster_id_to_sids: Dict[int, List[int]] = {}
    cluster_by_id: Dict[int, SemanticCluster] = {}
    for sid, cluster in sid_to_best.items():
        cid = id(cluster)
        cluster_id_to_sids.setdefault(cid, []).append(sid)
        cluster_by_id[cid] = cluster

    final: List[SemanticCluster] = []
    for cid, sids in cluster_id_to_sids.items():
        if len(sids) < 2:
            continue
        cluster = cluster_by_id[cid]
        surviving = [m for m in cluster.members if m.sid in sids]
        if len(surviving) >= 2:
            final.append(SemanticCluster(
                name=cluster.name,
                category=cluster.category,
                members=surviving,
                weight=cluster.weight,
                is_critical=cluster.is_critical,
            ))

    return final


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_semantic_clusters(shortcuts: List[Shortcut]) -> List[SemanticCluster]:
    """Build semantic clusters from the shortcut corpus.

    Combines multiple detection strategies:
    * action-stem directional / sequence tokens (e.g. left/right, copy/paste);
    * semantic keyword families (e.g. browser tabs, zoom, find, devtools);
    * cross-app exact-action matching (e.g. "Search" in Discord + Windows);
    * raw key pairs (PageUp/PageDown, arrow keys, etc.);
    * key pattern families (Ctrl+1..Ctrl+8, Win+1..Win+5).

    Returns a list of clusters.  Each cluster contains member SIDs, a canonical
    order, and canonical relative offsets.  Critical clusters are flagged for
    possible atomic group-move treatment.
    """
    by_sid = {s.sid: s for s in shortcuts}

    clusters: List[SemanticCluster] = []
    clusters.extend(_direction_and_sequence_clusters(shortcuts, by_sid))
    clusters.extend(_keyword_family_clusters(shortcuts, by_sid))
    clusters.extend(_cross_app_exact_clusters(shortcuts, by_sid))
    clusters.extend(_raw_key_clusters(shortcuts, by_sid))
    clusters.extend(_key_pattern_clusters(shortcuts, by_sid))

    return _deduplicate_clusters(clusters)
