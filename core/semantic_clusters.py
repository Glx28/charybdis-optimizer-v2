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
from core.norwegian_keys import RAW_COMPLETION_NORWEGIAN


# ---------------------------------------------------------------------------
# Directional / sequence / antonym vocabulary
# ---------------------------------------------------------------------------

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
    KeywordFamily("clipboard", ("cut", "copy", "paste", "clipboard"),
                  exclude_keywords=("line", "clipboard history"), min_members=2),
    # History (Undo/Redo).
    KeywordFamily("history", ("undo", "redo")),
    # Browser tabs.
    KeywordFamily("browser_tab", ("tab",),
                  direction_tokens=("new", "close", "reopen", "next", "previous"),
                  exclude_keywords=("switch to tab", "last tab")),
    # Browser navigation.
    KeywordFamily("browser_nav", ("back", "forward")),
    # Browser refresh/reload.
    KeywordFamily("browser_refresh", ("refresh", "reload", "hard refresh")),
    # Browser find.
    KeywordFamily("browser_find", ("find",), direction_tokens=("previous", "next", "on page"),
                  exclude_keywords=("my mouse",)),
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
    KeywordFamily("browser_panel", ("history", "downloads", "sidebar"),
                  exclude_keywords=("version history", "clipboard history")),
    # Browser fullscreen / view toggles.
    KeywordFamily("browser_view", ("fullscreen", "device toolbar")),
    # Window management (Windows).
    KeywordFamily("window_state", ("maximize", "minimize", "restore", "close window", "snap window"),
                  exclude_keywords=("window sets", "workspaces")),
    # Virtual desktops.
    KeywordFamily("virtual_desktop", ("virtual desktop", "desktop", "switch desktop")),
    # Monitor movement.
    KeywordFamily("monitor_move", ("monitor",), direction_tokens=("left", "right")),
    # Keep launchers in their actual contexts: PowerToys' two launch surfaces,
    # and Windows Search/Run. Browser address/search shortcuts have their own
    # family above; unrelated app search shortcuts do not join these groups.
    KeywordFamily("powertoys_launcher", ("launcher", "command palette"),
                  require_app="PowerToys"),
    KeywordFamily("windows_launcher", ("search", "run dialog"),
                  require_app="Windows 11"),
    # PowerToys feature groups. Keep unrelated utilities out of one app-wide
    # mega-cluster; the launcher pair and distinct tools form useful contexts.
    KeywordFamily("powertoys_visual_tools", ("color picker", "text extractor", "screen ruler"),
                  require_app="PowerToys"),
    KeywordFamily("powertoys_mouse_tools", ("mouse pointer crosshairs", "mouse highlighter", "find my mouse"),
                  require_app="PowerToys"),
    KeywordFamily("powertoys_workspace_tools", ("fancyzones", "workspaces"),
                  require_app="PowerToys"),
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
    # System settings / info.
    KeywordFamily("system_settings", ("settings", "system info")),
    # Input language.
    KeywordFamily("input_language", ("input language", "keyboard layout")),
    # Teams call controls.
    KeywordFamily("teams_call", ("mute", "deafen", "video", "raise hand", "lower hand")),
    # Raw completion keys.
    KeywordFamily("raw_completion", ()),
    KeywordFamily("modifier_keys", (), require_app="Modifier Keys"),
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


def _is_unmodified_arrow(sc: Shortcut) -> bool:
    return (not sc.modifiers
            and (sc.base_key or "").upper() in {
                "LEFTARROW", "RIGHTARROW", "UPARROW", "DOWNARROW",
            })


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
    """Curated directional pairs plus ordered sequence-word clusters."""
    seq_groups: Dict[Tuple[str, str], List[Tuple[int, str, int, float, float]]] = {}

    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        norm = _normalize_action(sc.action)
        if any(ig in norm for ig in IGNORED_ACTION_WORDS):
            continue

        seq_info = _extract_sequence(sc.action)
        if seq_info is not None:
            token, order, dx, dy, family = seq_info
            stem = _action_stem(sc.action, token) or "_"
            seq_groups.setdefault((stem, family), []).append((sc.sid, token, order, dx, dy))

    clusters: List[SemanticCluster] = _direction_clusters_for_tokens(shortcuts, by_sid)

    def _from_sequence_entries(key, entries):
        by_order: Dict[int, List[Tuple[int, str, int, float, float]]] = {}
        for sid, token, order, dx, dy in entries:
            by_order.setdefault(order, []).append((sid, token, order, dx, dy))
        if len(by_order) < 2:
            return None
        stem, family = key
        tokens = [items[0][1] for _, items in sorted(by_order.items())]
        name = _make_cluster_name(stem, "sequence", family, tokens)
        # Preserve alternate bindings for the same semantic action (for
        # example Ctrl+Y and Ctrl+Shift+Z for Redo). Keep aliases adjacent to
        # their sequence neighbors instead of dropping them from the ordered
        # contract or leaving them anywhere in a broad same-layer family.
        build_entries = []
        ordinal = min(by_order)
        for _semantic_order, items in sorted(by_order.items()):
            for sid, _token, _order, _dx, _dy in sorted(
                items, key=lambda item: (-by_sid[item[0]].importance, item[0]),
            ):
                build_entries.append((sid, ordinal, float(ordinal), 0.0))
                ordinal += 1
        return _build_cluster(name, "sequence", by_sid, build_entries, is_critical=True)

    for key, entries in seq_groups.items():
        cluster = _from_sequence_entries(key, entries)
        if cluster is not None:
            clusters.append(cluster)

    return clusters


def _keyword_family_clusters(
    shortcuts: List[Shortcut], by_sid: Dict[int, Shortcut],
    exclude_compact_sids: Optional[set] = None,
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
            if _skip_shortcut(sc) or _is_unmodified_arrow(sc):
                continue
            if family.name == "raw_completion":
                if sc.base_key in RAW_COMPLETION_NORWEGIAN and not sc.modifiers and not sc.is_l0_only:
                    matched.append(sc)
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

        # Do not make broad same-layer families pull ordered subgroups away
        # from their declared geometry. Keep compact pressure only for family
        # members that have no more specific ordered/pattern context.
        compact_members = [sc for sc in matched
                           if sc.sid not in (exclude_compact_sids or set())]
        entries = [(sc.sid, 0, 0.0, 0.0) for sc in compact_members]
        cluster = _build_cluster(f"family_{family.name}", "keyword_family", by_sid,
                                 entries, compactness=True)
        if cluster is not None:
            clusters.append(cluster)
        if family.direction_tokens:
            clusters.extend(_keyword_direction_clusters(matched, family, by_sid))

    return clusters


_CURATED_DIRECTION_PAIRS = (
    (("left", "right"), (0.0, 0.0), (1.0, 0.0)),
    (("top", "bottom"), (0.0, 0.0), (0.0, 1.0)),
    (("up", "down"), (0.0, 0.0), (0.0, 1.0)),
    (("above", "below"), (0.0, 0.0), (0.0, 1.0)),
    (("raise", "lower"), (0.0, 0.0), (0.0, 1.0)),
    (("previous", "next"), (0.0, 0.0), (1.0, 0.0)),
    (("prev", "next"), (0.0, 0.0), (1.0, 0.0)),
    (("back", "forward"), (0.0, 0.0), (1.0, 0.0)),
    (("backward", "forward"), (0.0, 0.0), (1.0, 0.0)),
    (("before", "after"), (0.0, 0.0), (1.0, 0.0)),
    (("earlier", "later"), (0.0, 0.0), (1.0, 0.0)),
    (("in", "out"), (0.0, 0.0), (1.0, 0.0)),
    (("increase", "decrease"), (0.0, 0.0), (0.0, 1.0)),
    (("horizontal", "vertical"), (0.0, 0.0), (1.0, 0.0)),
    (("open", "close"), (0.0, 0.0), (1.0, 0.0)),
    (("show", "hide"), (0.0, 0.0), (1.0, 0.0)),
    (("enable", "disable"), (0.0, 0.0), (1.0, 0.0)),
    (("minimize", "maximize"), (0.0, 0.0), (1.0, 0.0)),
    (("shrink", "expand"), (0.0, 0.0), (1.0, 0.0)),
    (("outdent", "indent"), (0.0, 0.0), (1.0, 0.0)),
    (("remove", "add"), (0.0, 0.0), (1.0, 0.0)),
    (("first", "last"), (0.0, 0.0), (1.0, 0.0)),
    (("home", "end"), (0.0, 0.0), (1.0, 0.0)),
    (("start", "stop"), (0.0, 0.0), (1.0, 0.0)),
    (("play", "pause"), (0.0, 0.0), (1.0, 0.0)),
    (("mute", "unmute"), (0.0, 0.0), (1.0, 0.0)),
    (("lock", "unlock"), (0.0, 0.0), (1.0, 0.0)),
)


def _direction_clusters_for_tokens(shortcuts, by_sid, allowed=None, prefix=""):
    """Make action-stem pairs only for curated, semantically compatible tokens."""
    allowed = set(allowed) if allowed is not None else None
    result = []
    for (left, right), left_offset, right_offset in _CURATED_DIRECTION_PAIRS:
        if allowed is not None and (left not in allowed or right not in allowed):
            continue
        buckets = {}
        for sc in shortcuts:
            if _skip_shortcut(sc) or _is_unmodified_arrow(sc):
                continue
            tokens = _tokenize(sc.action)
            token = left if left in tokens else (right if right in tokens else None)
            if token is None:
                continue
            stem = _action_stem(sc.action, token) or "_"
            pair = buckets.setdefault(stem, {})
            if token not in pair or sc.importance > pair[token].importance:
                pair[token] = sc
        for stem, pair in buckets.items():
            if left not in pair or right not in pair:
                continue
            left_sc, right_sc = pair[left], pair[right]
            # Preserve the user's physical direction when action wording is
            # abstract. PageUp/PageDown and Up/Down are vertical relations even
            # when the descriptions say Previous/Next (slides, tabs, items).
            left_key = re.sub(r"[^a-z0-9]", "", (left_sc.base_key or "").lower())
            right_key = re.sub(r"[^a-z0-9]", "", (right_sc.base_key or "").lower())
            vertical_context = (left_key, right_key) in {
                ("pageup", "pagedown"), ("uparrow", "downarrow"),
            }
            pair_left_offset, pair_right_offset = left_offset, right_offset
            if vertical_context:
                pair_left_offset, pair_right_offset = (0.0, 0.0), (0.0, 1.0)
            entries = [
                (left_sc.sid, 0, *pair_left_offset),
                (right_sc.sid, 1, *pair_right_offset),
            ]
            cluster = _build_cluster(
                _make_cluster_name(f"{prefix}{stem}", "direction"),
                "direction", by_sid, entries, is_critical=True,
            )
            if cluster is not None:
                result.append(cluster)
    return result


def _key_direction_clusters(shortcuts, by_sid):
    """Infer ordered pairs from matching modifier stacks and physical direction keys."""
    key_pairs = (
        (("left", "right"), ("Left", "Right"), (0.0, 0.0), (1.0, 0.0)),
        (("up", "down"), ("Up", "Down"), (0.0, 0.0), (0.0, 1.0)),
        (("home", "end"), ("Home", "End"), (0.0, 0.0), (1.0, 0.0)),
        (("page_up", "page_down"), ("PageUp", "PageDown"), (0.0, 0.0), (0.0, 1.0)),
        (("minus", "plus"), ("Minus", "Plus"), (0.0, 0.0), (1.0, 0.0)),
        (("less_than", "greater_than"), ("LessThan", "GreaterThan"), (0.0, 0.0), (1.0, 0.0)),
    )
    aliases = {
        "leftarrow": "Left", "left": "Left",
        "rightarrow": "Right", "right": "Right",
        "uparrow": "Up", "up": "Up",
        "downarrow": "Down", "down": "Down",
        "dash and underscore": "Minus", "equals and plus": "Plus",
        "comma and lessthan": "LessThan", "period and greaterthan": "GreaterThan",
    }
    result = []

    def actions_form_pair(first, second):
        first_tokens = set(_tokenize(first.action))
        second_tokens = set(_tokenize(second.action))
        for (left, right), _, _ in _CURATED_DIRECTION_PAIRS:
            if ((left in first_tokens and right in second_tokens)
                    or (right in first_tokens and left in second_tokens)):
                return True
        return False

    for pair_name, key_names, first_offset, second_offset in key_pairs:
        buckets = {}
        for sc in shortcuts:
            if _skip_shortcut(sc) or _is_unmodified_arrow(sc):
                continue
            base = (sc.base_key or sc.keys.rsplit("+", 1)[-1]).strip()
            normalized_base = aliases.get(base.lower(), base)
            if normalized_base not in key_names:
                continue
            modifier = "+".join(sc.modifiers) if sc.modifiers else "base"
            bucket = buckets.setdefault(modifier, {})
            if normalized_base not in bucket or sc.importance > bucket[normalized_base].importance:
                bucket[normalized_base] = sc
        for modifier, pair in buckets.items():
            first_key, second_key = key_names
            if first_key not in pair or second_key not in pair:
                continue
            if pair_name != ("home", "end") and not actions_form_pair(
                pair[first_key], pair[second_key],
            ):
                continue
            entries = [
                (pair[first_key].sid, 0, *first_offset),
                (pair[second_key].sid, 1, *second_offset),
            ]
            name = f"key_{modifier}_{pair_name[0]}_{pair_name[1]}".replace("+", "_")
            cluster = _build_cluster(name, "direction", by_sid, entries, is_critical=True)
            if cluster is not None:
                result.append(cluster)
    return result


def _keyword_direction_clusters(matched, family, by_sid):
    """Make action-stem pairs restricted by a family's declared direction tokens."""
    return _direction_clusters_for_tokens(
        matched, by_sid, allowed=family.direction_tokens, prefix=f"{family.name}_",
    )


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
    by_family: Dict[Tuple[str, str], List[Tuple[int, int, Shortcut]]] = {}
    for sc in shortcuts:
        if _skip_shortcut(sc):
            continue
        m = numeric_pattern.match(sc.keys)
        if m:
            mod, num = m.group(1), int(m.group(2))
            norm = _normalize_action(sc.action)
            number_pattern = re.compile(r"\b" + str(num) + r"\b")
            if number_pattern.search(norm):
                stem = number_pattern.sub(" ", norm)
                stem = re.sub(r"\s+", " ", stem).strip()
            elif mod == "Ctrl" and num == 9 and "last tab" in norm:
                stem = "switch to tab"
            else:
                # A modifier+digit alone is insufficient evidence for one
                # semantic sequence: Ctrl+0 (zoom reset) must not join tabs.
                continue
            by_family.setdefault((mod, stem), []).append((num, sc.sid, sc))

    for (mod, stem), items in by_family.items():
        if len(items) < 2:
            continue
        # Keep consecutive numeric runs of length >= 2.
        items_sorted = sorted(items, key=lambda x: x[0])
        run: List[Tuple[int, int, Shortcut]] = []

        def _run_entries(run_items):
            first_num = run_items[0][0]
            entries = []
            for num, sid, _ in run_items:
                ordinal = num - first_num
                if mod == "Ctrl" and stem == "switch to tab":
                    # Browser tab shortcuts form an ordered 3x3 navigation grid;
                    # a nine-position straight row does not fit the keywell.
                    dx, dy = ordinal % 3, ordinal // 3
                else:
                    # Taskbar slots and other numeric sequences stay in numeric
                    # order along a row.
                    dx, dy = ordinal, 0
                entries.append((sid, num, float(dx), float(dy)))
            return entries

        for it in items_sorted:
            if not run or it[0] == run[-1][0] + 1:
                run.append(it)
            else:
                if len(run) >= 2:
                    entries = _run_entries(run)
                    cluster = _build_cluster(f"pattern_{mod}_{stem.replace(' ', '_')}", "key_pattern", by_sid, entries)
                    if cluster is not None:
                        clusters.append(cluster)
                run = [it]
        if len(run) >= 2:
            entries = _run_entries(run)
            cluster = _build_cluster(f"pattern_{mod}_{stem.replace(' ', '_')}", "key_pattern", by_sid, entries)
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
    """Remove duplicate definitions while preserving overlapping subgroups."""
    chosen = {}
    for cluster in clusters:
        if len(cluster.members) < 2:
            continue
        member_set = tuple(sorted({m.sid for m in cluster.members}))
        has_relation = any(abs(m.dx) > 0.01 or abs(m.dy) > 0.01 for m in cluster.members)
        rank = (has_relation, cluster.is_critical, _cluster_priority(cluster))
        existing = chosen.get(member_set)
        if existing is None:
            chosen[member_set] = (rank, cluster)
            continue
        existing_cluster = existing[1]
        candidate_is_physical = cluster.name.startswith("key_")
        existing_is_physical = existing_cluster.name.startswith("key_")
        if candidate_is_physical != existing_is_physical:
            signature = lambda item: tuple(sorted(
                (m.sid, round(m.dx, 3), round(m.dy, 3)) for m in item.members
            ))
            candidate_signature = signature(cluster)
            existing_signature = signature(existing_cluster)
            # Physical key placement overrides abstract action ordering only
            # when the two infer different geometries. Preserve the clearer
            # action-derived name when both agree (e.g. LeftArrow/RightArrow).
            if candidate_is_physical and candidate_signature != existing_signature:
                chosen[member_set] = (rank, cluster)
            continue
        if rank > existing[0]:
            chosen[member_set] = (rank, cluster)
    return sorted((item[1] for item in chosen.values()), key=lambda c: (c.name, tuple(c.member_sids)))


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
    clusters.extend(_key_direction_clusters(shortcuts, by_sid))
    clusters.extend(_key_pattern_clusters(shortcuts, by_sid))
    ordered_sids = set()
    for cluster in clusters:
        if any(abs(member.dx) > 0.01 or abs(member.dy) > 0.01
               for member in cluster.members):
            ordered_sids.update(member.sid for member in cluster.members)
    clusters.extend(_keyword_family_clusters(shortcuts, by_sid, ordered_sids))
    clusters.extend(_cross_app_exact_clusters(shortcuts, by_sid))
    clusters.extend(_raw_key_clusters(shortcuts, by_sid))

    return _deduplicate_clusters(clusters)
