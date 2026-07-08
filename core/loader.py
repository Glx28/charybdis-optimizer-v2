"""Data loader: reads v1 data files and builds v2 data structures."""
import json
import os
import re
from dataclasses import replace
from typing import Optional, Tuple, List, Dict
import numpy as np

from core import Position, Shortcut, Layout, UsageData, LayerAccess
from core.norwegian_keys import canonical_hid_parameter, parse_shortcut_keys_norwegian


RAW_ARROW_BASE_KEYS = {"LeftArrow", "RightArrow", "UpArrow", "DownArrow"}


RAW_KEY_ALIASES = {
    "escape": "escape",
    "esc": "escape",
    "delete": "delete",
    "bksp": "delete",
    "backspace": "delete",
    "tab": "tab",
    "space": "spacebar",
    "spacebar": "spacebar",
    "enter": "returnenter",
    "return": "returnenter",
    "return enter": "returnenter",
    "leftshift": "leftshift",
    "rightshift": "rightshift",
    "shift": "leftshift",
    "leftcontrol": "leftcontrol",
    "rightcontrol": "rightcontrol",
    "ctrl": "leftcontrol",
    "leftalt": "leftalt",
    "rightalt": "rightalt",
    "left gui": "leftgui",
    "leftgui": "leftgui",
    "comma": ",",
    "comma and lessthan": ",",
    "period": ".",
    "period and greaterthan": ".",
    "forwardslash": "/",
    "forwardslash and questionmark": "/",
    "backslash": "\\",
    "backslash and pipe": "\\",
    "semicolon": ";",
    "semicolon and colon": ";",
    "left apos": "'",
    "left apos and double": "'",
    "apostrophe": "'",
    "left brace": "[",
    "right brace": "]",
}

L0_FROZEN_THUMB_RAW_KEYS = {"spacebar", "returnenter"}
NON_GROUPABLE_KEYS = {"ScrollUp", "ScrollDown"}
NON_EXPORTABLE_SEQUENCE_KEYS = {"Ctrl+K S", "gg", "gi", "yy"}
MOUSE_CLICK_BASE_KEYS = {"Click", "Left Click", "Right Click", "Middle Click"}
MODIFIER_PREFIXES = ("Ctrl+", "Alt+", "Shift+", "Win+", "Meta+", "Cmd+")


def _normalize_raw_key_id(value: str) -> Optional[str]:
    """Normalize a raw no-modifier key to a stable id."""
    if value is None:
        return None
    clean = str(value).strip()
    if not clean:
        return None
    clean = clean.replace("Keyboard ", "").replace("keyboard ", "")
    clean = clean.split(" and ")[0] if " and " in clean and clean[:1].isdigit() else clean
    lowered = clean.lower().strip()
    if lowered in RAW_KEY_ALIASES:
        return RAW_KEY_ALIASES[lowered]
    if len(clean) == 1:
        return clean.lower()
    if clean.isdigit():
        return clean
    if re.fullmatch(r"f\d{1,2}", lowered):
        return lowered
    return None


def _permanent_l0_raw_keys(layout_data: dict) -> Dict[str, str]:
    """Return raw keys permanently present on L0 (frozen finger + frozen thumb keys)."""
    result = {}
    l0_frozen = layout_data.get("l0_frozen", {})
    for coord, kd in l0_frozen.items():
        behavior = kd.get("behavior", "").lower()
        if "key press" not in behavior and "key" not in behavior:
            continue
        if kd.get("modifiers"):
            continue
        param = kd.get("parameter", "")
        key_id = _normalize_raw_key_id(param)
        if key_id:
            result[key_id] = kd.get("label") or param or key_id
    return result


def _parse_layer_from_behavior(label: str, behavior: str, parameter: str) -> Optional[int]:
    """Extract target layer number from explicit firmware metadata only.

    Only L0 and L7 have predetermined roles. All other layers are assigned by
    evolution. Behaviors must use coach_lN_hold/toggle/lock (numeric) to
    reference non-L0/non-L7 layers — function names like coach_mouse_lock are
    not valid and will return None (ignored as non-layer-access).
    """
    if parameter:
        if parameter.isdigit():
            return int(parameter)
        m = re.search(r'Layer::(\d+)', parameter)
        if m:
            return int(m.group(1))

    # Numeric coach names — the only valid non-L0/non-L7 layer references.
    m = re.search(r'coach_l(\d+)_(?:hold|toggle|lock)', behavior or "", re.IGNORECASE)
    if m:
        return int(m.group(1))

    # coach_travel_off kept for legacy compatibility (= coach_base = return to L0)
    if 'coach_travel_off' in behavior:
        return 0
    if 'coach_base' in behavior:
        return 0
    if 'coach_game_lock' in behavior or 'coach_game_hold' in behavior:
        return 7  # L7 is the game layer — the only predetermined non-base layer.

    return None


def _is_momentary_access(behavior: str) -> bool:
    """Determine if a layer access key requires holding (momentary)."""
    # Explicit non-momentary patterns first
    if 'toggle' in behavior.lower():
        return False
    if 'lock' in behavior.lower():
        return False
    if 'base' in behavior.lower():
        return False  # return to base keys
    if 'travel_off' in behavior.lower():
        return False

    # Momentary patterns
    if 'hold' in behavior.lower():
        return True
    if 'momentary' in behavior.lower():
        return True
    if 'coach_l1_hold' in behavior or 'coach_l2_hold' in behavior or 'coach_l3_hold' in behavior or 'coach_l4_hold' in behavior:
        return True

    return False


def _hand_from_x(x: float) -> str:
    """Determine hand from x coordinate."""
    return "left" if x < 6 else "right"


def _access_shortcut_key(source_layer: int, target_layer: int, is_momentary: bool, label: str) -> str:
    mode = "hold" if is_momentary else "toggle"
    clean = re.sub(r"[^A-Za-z0-9]+", "_", label or f"L{target_layer}").strip("_") or f"L{target_layer}"
    return f"@access:L{source_layer}->L{target_layer}:{mode}:{clean}"


def _is_plain_keypress_shortcut(keys: str, sc_data: dict) -> bool:
    """True when the shortcut can be represented as one Studio binding.

    Mouse-click shortcuts are valid, but they must later export as Mouse Key
    Press bindings (MB1/MB2/MB3), not as keyboard parameters named Click.
    Scroll wheel events and text sequences are usage signals only; they are not
    assignable one-key ZMK Studio keypresses.
    """
    clean = str(keys or "").strip()
    category = str(sc_data.get("category", "")).lower()
    action = str(sc_data.get("action", "")).lower()
    behavior = str(sc_data.get("behavior", "")).lower()
    parameter = str(sc_data.get("parameter", "")).lower()
    if _is_structural_system_key(clean, behavior, parameter, action):
        return False

    modifiers, parsed_base = parse_shortcut_keys_norwegian(clean)
    base_key = canonical_hid_parameter(sc_data.get("base_key", "")) or parsed_base

    if clean in NON_EXPORTABLE_SEQUENCE_KEYS:
        return False
    # Multi-stroke chords such as "Ctrl+K Ctrl+F" are useful usage signals,
    # but they are not one ZMK Studio binding.  They must not occupy prime
    # optimizer slots and then export as transparent.
    if " " in clean and any(prefix in clean for prefix in MODIFIER_PREFIXES):
        return False
    if clean in NON_GROUPABLE_KEYS or base_key in NON_GROUPABLE_KEYS:
        return False
    if "scroll" in action and (clean in NON_GROUPABLE_KEYS or base_key in NON_GROUPABLE_KEYS):
        return False

    if base_key in MOUSE_CLICK_BASE_KEYS:
        return True

    # Multi-step text/editor sequences are useful usage data, but they are not
    # one physical key binding. Modifier chords still pass because they contain +.
    if "vimium" in category and "+" not in clean and len(clean) > 1:
        return False
    if not modifiers and re.fullmatch(r"[A-Za-z]{2,}", clean):
        # Allow solo modifier keys (LeftAlt, LeftCtrl, etc.) and arrow keys — these
        # are valid physical keys, not text sequences.
        if _normalize_raw_key_id(clean) is None and clean not in RAW_ARROW_BASE_KEYS:
            return False

    return True


def _is_structural_system_key(label: str, behavior: str, parameter: str, action: str = "") -> bool:
    """True for canonical system controls that must not enter the evolvable genome."""
    text = f"{label} {behavior} {parameter} {action}".lower()
    return (
        "bluetooth" in text
        or "bt_sel" in text
        or "output selection" in text
        or "out_sel" in text
    )


def _is_scroll_mode_access(label: str, behavior: str, parameter: str) -> bool:
    """True for access keys that switch the trackball from pointer to wheel mode."""
    text = f"{label} {behavior} {parameter}".lower()
    return "scroll" in text


def _build_layer_access() -> List[LayerAccess]:
    """Layer access is fully evolved; there is no static access list."""
    return []


def load_layout(path: str) -> Tuple[List[Position], np.ndarray, List[LayerAccess]]:
    """Load layout.json and build positions with frozen info."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = []
    frozen = []

    physical_grid = data.get("physical_grid", {})
    position_metas = physical_grid.get("positions", [])
    n_layers = data.get("n_layers", 11)

    finger_map = {
        "thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4,
        "far_pinky": 4, "index_stretch": 1,
    }
    # Fallback only for old/minimal test fixtures missing explicit effort.
    # Real keyboard data must provide per-coordinate effort in layout.json.
    effort_map = {
        "top": 2.0, "upper": 1.0, "middle": 0.0, "home": 0.0,
        "lower": 1.0, "bottom": 1.0, "thumb": 0.0, "thumb2": 1.0,
    }

    l0_frozen = data.get("l0_frozen", {})

    for layer_num in range(n_layers):
        for pos_meta in position_metas:
            x = float(pos_meta.get("x", 0))
            y = float(pos_meta.get("y", 0))
            finger_str = pos_meta.get("finger", "index")
            finger = finger_map.get(finger_str, 1)
            row_type = pos_meta.get("row_type", "middle")
            # Prefer explicit per-coordinate effort from layout.json. The row_type
            # fallback exists only for old/minimal test fixtures.
            effort = float(pos_meta.get("effort", effort_map.get(row_type, 1.0)))
            is_thumb = pos_meta.get("zone") == "thumb" or finger == 0

            coord = f"{int(x)}:{int(y)}"
            if layer_num == 0:
                is_frozen = coord in l0_frozen
            elif layer_num == 7:
                is_frozen = True
            else:
                is_frozen = False

            pos = Position(
                gene_idx=len(positions),
                layer=layer_num,
                x=x,
                y=y,
                hand=pos_meta.get("hand", "left"),
                finger=finger,
                effort=effort,
                is_thumb=is_thumb,
                is_frozen=is_frozen,
                row=0,
                col=len(positions) % len(position_metas),
            )
            positions.append(pos)
            frozen.append(is_frozen)

    frozen_mask = np.array(frozen, dtype=bool)
    return positions, frozen_mask, _build_layer_access()


def load_shortcuts(path: str, layout_data: Optional[dict] = None) -> List[Shortcut]:
    """Load app_shortcut_scores.json and build Shortcut objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shortcuts = []
    seen_keys = set()
    permanent_l0_raw = _permanent_l0_raw_keys(layout_data or {})
    scroll_total = 0
    if layout_data:
        scroll_total = int((layout_data.get("_usage_stats") or {}).get("scroll_total", 0) or 0)

    for key_id, label in sorted(permanent_l0_raw.items()):
        shortcuts.append(Shortcut(
            sid=len(shortcuts),
            keys=f"_base_{key_id}",
            action="Base typing key for the permanent L0 layout.",
            app="Base",
            importance=0.0,
            category="base",
            modifiers=tuple(),
            base_key=label,
            is_l0_only=True,
            complexity=1,
            preferred_hand="either",
        ))

    # Generate all possible layer-access shortcuts programmatically.
    # No canonical bias — evolution places these freely anywhere.
    n_layers = (layout_data or {}).get("n_layers", 11) if layout_data else 11
    access_seen: set = set()
    for tgt in range(n_layers):
        if tgt == 0:
            keys = "@access:L0:return"
            if keys not in access_seen:
                access_seen.add(keys)
                shortcuts.append(Shortcut(
                    sid=len(shortcuts), keys=keys,
                    action="Return to Base", app="Layer Access",
                    importance=10.0, category="layer_access",
                    modifiers=tuple(), base_key="L0",
                    is_capability=True, is_l0_only=False, complexity=1,
                    preferred_hand="either", is_layer_access=True,
                    access_target_layer=0, access_is_momentary=False,
                ))
            continue

        is_game = (tgt == 7)
        base_importance = 22.0 if is_game else 14.0

        # Momentary hold
        keys_hold = f"@access:L{tgt}:hold"
        if keys_hold not in access_seen:
            access_seen.add(keys_hold)
            shortcuts.append(Shortcut(
                sid=len(shortcuts), keys=keys_hold,
                action=f"Momentary Layer {tgt}", app="Layer Access",
                importance=base_importance, category="layer_access",
                modifiers=tuple(), base_key=f"L{tgt}",
                is_capability=True, is_l0_only=False, complexity=1,
                preferred_hand="either", is_layer_access=True,
                access_target_layer=tgt, access_is_momentary=True,
            ))

        # Toggle access is required for every generated/frozen target layer,
        # including L7. L7 content is frozen, but its access mode is evolvable.
        keys_toggle = f"@access:L{tgt}:toggle"
        if keys_toggle not in access_seen:
            access_seen.add(keys_toggle)
            shortcuts.append(Shortcut(
                sid=len(shortcuts), keys=keys_toggle,
                action=f"Toggle Layer {tgt}", app="Layer Access",
                importance=base_importance - 2.0, category="layer_access",
                modifiers=tuple(), base_key=f"L{tgt}",
                is_capability=True, is_l0_only=False, complexity=1,
                preferred_hand="either", is_layer_access=True,
                access_target_layer=tgt, access_is_momentary=False,
            ))

        # Scroll-mode hold: "scroll" in keys triggers shortcut_scroll_mode_access in kernel.
        if not is_game:
            keys_scroll = f"@scroll:L{tgt}:hold"
            if keys_scroll not in access_seen:
                access_seen.add(keys_scroll)
                scroll_imp = 12.0  # second after MB1 (15.0), above MB2 (9.0)
                shortcuts.append(Shortcut(
                    sid=len(shortcuts), keys=keys_scroll,
                    action=f"Scroll Mode Layer {tgt}", app="Layer Access",
                    importance=scroll_imp, category="layer_access",
                    modifiers=tuple(), base_key=f"Scroll_L{tgt}",
                    is_capability=True, is_l0_only=False, complexity=1,
                    preferred_hand="right", is_layer_access=True,
                    access_target_layer=tgt, access_is_momentary=True,
                ))

    raw_arrow_by_base: Dict[str, int] = {}

    for app_data in data.get("apps", []):
        app_name = app_data.get("name", "unknown")
        for sc_data in app_data.get("shortcuts", []):
            keys = sc_data.get("keys", "")
            if keys in seen_keys:
                continue
            if not _is_plain_keypress_shortcut(keys, sc_data):
                continue
            seen_keys.add(keys)

            # Extract canonical HID base key from keys if not provided. Display
            # symbols are Norwegian OS-layout results and must not become Studio
            # parameters.
            base_key = canonical_hid_parameter(sc_data.get("base_key", ""))
            if not base_key:
                _, base_key = parse_shortcut_keys_norwegian(keys)

            # Extract modifiers from keys. Direct mouse-click names are mouse
            # actions, not keyboard words/modifiers. Modifier+Click remains one
            # mouse binding with held modifiers.
            modifiers, parsed_base = parse_shortcut_keys_norwegian(keys)
            if keys in MOUSE_CLICK_BASE_KEYS:
                modifiers = []
                base_key = keys
            elif parsed_base:
                base_key = parsed_base

            raw_key_id = _normalize_raw_key_id(base_key) if not modifiers and "+" not in keys else None
            if raw_key_id is not None and raw_key_id in permanent_l0_raw:
                continue

            if not modifiers and base_key in RAW_ARROW_BASE_KEYS:
                existing_idx = raw_arrow_by_base.get(base_key)
                if existing_idx is not None:
                    existing = shortcuts[existing_idx]
                    new_importance = float(sc_data.get("importance", 5.0))
                    if new_importance > existing.importance:
                        shortcuts[existing_idx] = replace(
                            existing,
                            keys=base_key,
                            action=sc_data.get("action", existing.action),
                            app=app_name,
                            importance=new_importance,
                            category=sc_data.get("category", existing.category),
                            preferred_hand=sc_data.get("preferred_hand", existing.preferred_hand),
                        )
                    continue

            sc = Shortcut(
                sid=len(shortcuts),
                keys=keys,
                action=sc_data.get("action", ""),
                app=app_name,
                importance=float(sc_data.get("importance", 5.0)),
                category=sc_data.get("category", "general"),
                modifiers=tuple(modifiers),
                base_key=base_key,
                is_capability=sc_data.get("is_capability", False),
                is_l0_only=sc_data.get("is_l0_only", False),
                complexity=int(sc_data.get("complexity", 1)),
                preferred_hand=sc_data.get("preferred_hand", "either"),
            )
            shortcuts.append(sc)
            if not modifiers and base_key in RAW_ARROW_BASE_KEYS:
                raw_arrow_by_base[base_key] = len(shortcuts) - 1

    # Ensure the Norwegian raw completion-key family is always present as
    # assignable capabilities, even if the usage logger has never recorded an
    # unmodified PageUp/Home/End/etc.  Without these base shortcuts the
    # completion cluster cannot form because there is nothing to cluster.
    _ensure_raw_completion_shortcuts(shortcuts)

    return shortcuts


def _ensure_raw_completion_shortcuts(shortcuts: List[Shortcut]):
    """Add missing unmodified raw completion-family keys synthesized from demand.

    The Norwegian extra-key family: the 5 physical keys that differ between
    Norwegian and US International keyboards, using US International HID names.
    Every family member must have a bare (no-modifier) shortcut so the optimizer
    can place it as part of the atomic group.  Members without demand are still
    synthesised so the group always has exactly 5 slots to move as a unit.
    """
    family = {
        "Dash and Underscore",
        "Equals and Plus",
        "Grave Accent and Tilde",
        "Right Brace",
        "Backslash and Pipe",
    }
    present_params = {
        sc.base_key for sc in shortcuts
        if sc.base_key in family and not sc.modifiers and sc.app != "Base"
    }
    seen_keys = {sc.keys.upper() for sc in shortcuts}

    default_family_keys = {
        "Dash and Underscore": "-",
        "Equals and Plus": "=",
        "Grave Accent and Tilde": "`",
        "Right Brace": "]",
        "Backslash and Pipe": "\\",
    }
    # Synthesise ALL missing family members (not just those with demand) so the
    # 5-key group always has complete coverage for the atomic group mutation.
    for param in sorted(family - present_params):
        keys = default_family_keys[param]
        keys = _unique_keys(keys, seen_keys)
        _append_raw_completion_shortcut(shortcuts, keys, param)
        present_params.add(param)


def _unique_keys(keys: str, seen_keys: set) -> str:
    """Return a keys string that does not collide with existing shortcut keys."""
    original = keys
    counter = 1
    while keys.upper() in seen_keys:
        counter += 1
        keys = f"{original} ({counter})"
    seen_keys.add(keys.upper())
    return keys


def _append_raw_completion_shortcut(shortcuts: List[Shortcut], keys: str, param: str):
    """Append a low-importance raw completion capability to the shortcut list."""
    shortcuts.append(Shortcut(
        sid=len(shortcuts),
        keys=keys,
        action=f"Raw {param} completion key.",
        app="Raw Keys",
        importance=3.0,
        category="raw_completion",
        modifiers=tuple(),
        base_key=param,
        is_capability=True,
        complexity=1,
        preferred_hand="either",
    ))


def load_usage_stats(path: str) -> Optional[UsageData]:
    """Load usage_stats.json if it exists."""
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shortcut_sequences = data.get("shortcut_sequences") or data.get("sequences", {})
    shortcut_workflows = data.get("shortcut_workflows") or data.get("workflows", {})
    chains = data.get("chains", {})
    if not chains and shortcut_workflows:
        chains = shortcut_workflows

    return UsageData(
        sequences=shortcut_sequences,
        chains=chains,
        workflows=shortcut_workflows,
        shortcut_sequences=shortcut_sequences,
        shortcut_workflows=shortcut_workflows,
        app_sequences=data.get("app_sequences", {}),
        app_workflows=data.get("app_workflows", {}),
        shortcuts=data.get("shortcuts", {}),
        mouse_session_shortcuts=data.get("mouse_session_shortcuts", {}),
        mouse_clicks=data.get("mouse_clicks", {}),
        scroll_total=int(data.get("scroll_total", 0) or 0),
        scroll_by_layer=data.get("scroll_by_layer", {}),
        raw_completion_keys=data.get("raw_completion_keys", {}),
        raw_completion_total=int(data.get("raw_completion_total", 0) or 0),
        by_layer_shortcut=data.get("by_layer_shortcut", {}),
        layer_shortcuts=data.get("layer_shortcuts", {}),
        by_app=data.get("by_app", {}),
        app_time_seconds=data.get("app_time_seconds", {}),
        total_events=data.get("total_events", 0),
        blind_spots=data.get("blind_spots", {}),
    )


def build_frozen_genome(layout_data: dict, positions: List[Position], shortcuts: List[Shortcut]) -> np.ndarray:
    """Build genome with only frozen positions assigned. Mutable positions are -1.

    L0 frozen positions get their base key sids.
    L7 is frozen firmware/export structure, but it is not represented as
    assigned shortcut SIDs during training.
    All non-L0 positions are left -1 for random assignment by the sampler.
    """
    n_pos = len(positions)
    genome = np.full(n_pos, -1, dtype=np.int32)

    base_keys_to_sid = {s.keys: s.sid for s in shortcuts if s.keys.startswith("_base_")}

    # L0 frozen positions
    l0_frozen = layout_data.get("l0_frozen", {})
    for pos in positions:
        if pos.layer != 0 or not pos.is_frozen:
            continue
        coord = f"{int(pos.x)}:{int(pos.y)}"
        kd = l0_frozen.get(coord, {})
        param = kd.get("parameter", "")
        key_id = _normalize_raw_key_id(param)
        if key_id:
            sid = base_keys_to_sid.get(f"_base_{key_id}")
            if sid is not None:
                genome[pos.gene_idx] = sid

    return genome


def _extract_base_key_id(param: str, behavior: str, modifiers: List[str]) -> Optional[str]:
    """Extract base key identifier for _base_* shortcut lookup."""
    p = param.lower()
    b = behavior.lower()

    # Coach behaviors
    coach_keys = {
        "coach_l1_hold", "coach_l2_hold", "coach_l3_hold", "coach_l4_hold",
        "coach_game_lock", "coach_game_hold", "coach_base",
        "coach_travel_off", "coach_recover_base",
    }
    if b in coach_keys:
        return b

    # Layer control
    if behavior in ("Momentary Layer", "Toggle Layer", "To Layer"):
        return f"{b.replace(' ', '_')}_{param}"

    # Mouse buttons
    if "mouse key press" in b:
        mb_match = re.search(r"(?:select:\s*)?mb\s*([1-5])", p + " " + behavior, re.IGNORECASE)
        if mb_match:
            return f"select:mb{mb_match.group(1)}"
        return p if p else None

    # Bluetooth/output
    if any(kw in b for kw in ["bluetooth", "output selection"]):
        return p if p else None

    # Modifier keys
    for kw in ["spacebar", "return enter", "tab", "escape", "delete",
               "leftshift", "rightshift", "leftcontrol", "rightcontrol",
               "leftalt", "rightalt", "left gui"]:
        if kw in p:
            return _normalize_raw_key_id(kw) or kw

    # Single raw key (no modifiers)
    if not modifiers:
        raw_id = _normalize_raw_key_id(p)
        if raw_id is not None:
            return raw_id

    # F-keys
    f_match = re.search(r'\bf(\d+)\b', p)
    if f_match:
        return f"f{f_match.group(1)}"

    return None


def build_layout(data_dir: str, config: dict = None) -> Layout:
    """Build a Layout from data files in ``data_dir``.

    Reads layout.json (positions + L0/L7 frozen firmware structure) and
    app_shortcut_scores.json (shortcut corpus). No placement seed is loaded for
    mutable layers or L7.
    """
    layout_path = os.path.join(data_dir, "layout.json")
    with open(layout_path, "r", encoding="utf-8") as f:
        layout_data = json.load(f)

    usage = load_usage_stats(os.path.join(data_dir, "usage_stats.json"))
    layout_data["_usage_stats"] = {
        "scroll_total": usage.scroll_total if usage else 0,
        "scroll_by_layer": usage.scroll_by_layer if usage else {},
    }

    positions, frozen_mask, layer_access = load_layout(layout_path)
    shortcuts = load_shortcuts(os.path.join(data_dir, "app_shortcut_scores.json"), layout_data)

    # Apply importance overrides from config
    if config:
        overrides = config.get("shortcut_importance_overrides", {})
        if overrides:
            updated = []
            for sc in shortcuts:
                override_key = None
                if sc.keys in overrides:
                    override_key = sc.keys
                elif sc.base_key in overrides:
                    override_key = sc.base_key
                if override_key is not None:
                    updated.append(Shortcut(
                        sid=sc.sid,
                        keys=sc.keys,
                        action=sc.action,
                        app=sc.app,
                        importance=float(overrides[override_key]),
                        category=sc.category,
                        modifiers=sc.modifiers,
                        base_key=sc.base_key,
                        is_capability=sc.is_capability,
                        is_l0_only=sc.is_l0_only,
                        complexity=sc.complexity,
                        preferred_hand=sc.preferred_hand,
                        primary_app=sc.primary_app,
                        app_demand=sc.app_demand,
                        is_layer_access=sc.is_layer_access,
                        access_target_layer=sc.access_target_layer,
                        access_is_momentary=sc.access_is_momentary,
                    ))
                else:
                    updated.append(sc)
            shortcuts = updated

    # Build layer_to_indices mapping
    layer_to_indices = {}
    for layer in set(p.layer for p in positions):
        indices = [p.gene_idx for p in positions if p.layer == layer]
        layer_to_indices[layer] = np.array(indices, dtype=np.int32)

    # Frozen-only genome — mutable positions are -1 for random sampler assignment.
    genome = build_frozen_genome(layout_data, positions, shortcuts)

    # Discover dynamic groups from usage data
    dynamic_groups = _discover_dynamic_groups(usage, shortcuts) if usage else []

    layout = Layout(
        genome=genome,
        positions=tuple(positions),
        shortcuts=tuple(shortcuts),
        frozen_mask=frozen_mask,
        layer_to_indices=layer_to_indices,
        usage_data=usage if usage else UsageData(),
        layer_access=tuple(layer_access),
        dynamic_groups=tuple(dynamic_groups),
    )

    return layout


def _discover_dynamic_groups(usage: UsageData, shortcuts: List[Shortcut]) -> List[Dict]:
    """Discover dynamic groups from usage sequences and chains.

    Ported from v1 evolve/representation.py discover_dynamic_groups().
    """
    sid_lookup = {s.keys: s.sid for s in shortcuts if s.keys not in NON_GROUPABLE_KEYS}
    pair_weights = {}

    # Build pair weights from usage sequences
    for seq_key, data in usage.sequences.items():
        count = data.get("count", 0) if isinstance(data, dict) else data
        if count < 2:
            continue
        parts = seq_key.split(" -> ")
        if len(parts) != 2:
            continue
        sid_a = sid_lookup.get(parts[0])
        sid_b = sid_lookup.get(parts[1])
        if sid_a is not None and sid_b is not None:
            key = tuple(sorted([sid_a, sid_b]))
            pair_weights[key] = pair_weights.get(key, 0) + count

    if not pair_weights:
        return []

    max_w = max(pair_weights.values())
    if max_w <= 0:
        return []

    groups = []
    used_sids = set()

    # Chain-derived groups
    seen_sid_sets = set()
    for chain_key, chain_data in usage.chains.items():
        parts = chain_key.split(" -> ")
        count = chain_data.get("count", 0) if isinstance(chain_data, dict) else chain_data
        if count < 3 or len(parts) < 2:
            continue
        chain_sids = []
        for p in parts:
            sid = sid_lookup.get(p)
            if sid is not None and sid not in chain_sids:
                chain_sids.append(sid)
        if len(chain_sids) < 2:
            continue
        if any(s in used_sids for s in chain_sids):
            continue
        sid_set = tuple(sorted(chain_sids))
        if sid_set in seen_sid_sets:
            continue
        seen_sid_sets.add(sid_set)
        name = f"chain_{'_'.join(shortcuts[s].keys for s in chain_sids[:3])}"
        groups.append({
            "name": name,
            "sids": chain_sids,
            "weight": 1.0,
            "protected": True,
            "dynamic": True,
        })
        for s in chain_sids:
            used_sids.add(s)

    # Pair-derived groups from sequence weights
    sorted_pairs = sorted(pair_weights.items(), key=lambda x: -x[1])
    threshold = 0.15
    for (sid_a, sid_b), w in sorted_pairs:
        norm_w = w / max_w
        if norm_w < threshold:
            break
        if sid_a in used_sids or sid_b in used_sids:
            continue
        sa = shortcuts[sid_a]
        sb = shortcuts[sid_b]
        groups.append({
            "name": f"dynamic_{sa.keys}_{sb.keys}",
            "sids": [sid_a, sid_b],
            "weight": norm_w,
            "protected": True,
            "dynamic": True,
        })
        used_sids.add(sid_a)
        used_sids.add(sid_b)

    return groups
