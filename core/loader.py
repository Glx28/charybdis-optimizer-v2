"""Data loader: reads v1 data files and builds v2 data structures."""
import json
import os
import re
from typing import Optional, Tuple, List, Dict
import numpy as np

from core import Position, Shortcut, Layout, UsageData, LayerAccess


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


def _permanent_l0_raw_keys(canonical_data: dict) -> Dict[str, str]:
    """Return raw keys permanently present on L0 rows y=0..3 plus frozen thumb exceptions."""
    permanent = {}
    l0_keys = canonical_data.get("layers", {}).get("0", {}).get("keys", {})
    for key_data in l0_keys.values():
        if key_data.get("behavior", "").lower() != "key press":
            continue
        if key_data.get("modifiers"):
            continue
        key_id = _normalize_raw_key_id(key_data.get("parameter", ""))
        if key_id is None:
            key_id = _normalize_raw_key_id(key_data.get("label", ""))
        try:
            y = float(key_data.get("y", 0))
        except (TypeError, ValueError):
            continue
        is_main_l0_key = 0 <= y <= 3
        is_frozen_thumb_exception = key_id in L0_FROZEN_THUMB_RAW_KEYS
        if not is_main_l0_key and not is_frozen_thumb_exception:
            continue
        if key_id is not None:
            permanent[key_id] = key_data.get("label") or key_data.get("parameter") or key_id
    return permanent


def _parse_layer_from_behavior(label: str, behavior: str, parameter: str) -> Optional[int]:
    """Extract target layer number from a layer access key."""
    # Check parameter first (most reliable)
    if parameter:
        if parameter.isdigit():
            return int(parameter)
        m = re.search(r'Layer::(\d+)', parameter)
        if m:
            return int(m.group(1))
    
    # Check behavior patterns
    if 'coach_l1_hold' in behavior:
        return 1
    if 'coach_l2_hold' in behavior:
        return 2
    if 'coach_l3_hold' in behavior:
        return 3
    if 'coach_l4_hold' in behavior:
        return 4
    if 'coach_travel_toggle' in behavior:
        return 8  # Speed/Travel layer
    if 'coach_travel_off' in behavior:
        return 0  # return to base
    if 'coach_base' in behavior:
        return 0  # return to base
    if 'coach_game_lock' in behavior:
        return 7  # Game/RPG layer
    if 'coach_mouse_lock' in behavior:
        return 2  # Mouse QoL layer
    
    # Check label for hints
    if 'Nav' in label:
        return 1
    if 'Mouse' in label:
        return 2
    if 'Window' in label:
        return 3
    if 'System' in label:
        return 4
    if 'Code' in label:
        return 5
    if 'Scroll' in label:
        return 6
    if 'Speed' in label or 'Travel' in label:
        return 8
    if 'DMS' in label or 'M-Files' in label:
        return 9
    if 'Excel' in label:
        return 10
    
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


def _build_layer_access(canonical_data: dict) -> List[LayerAccess]:
    """Build layer access mappings from canonical.json.
    
    For each target layer, traces the access chain from L0 and determines
    which thumb(s) are occupied while the layer is active.
    
    Thumb occupancy rule: if a layer is accessed via a momentary hold on a thumb,
    that thumb cannot be used to press other keys while the layer is active.
    Toggle/lock accesses do NOT occupy the thumb (the user can release the
    previous momentary after toggling).
    """
    access_list = []
    
    # Step 1: Find direct access keys on L0
    l0_keys = canonical_data.get('layers', {}).get('0', {}).get('keys', {})
    for coord, key_data in l0_keys.items():
        behavior = key_data.get('behavior', '')
        parameter = key_data.get('parameter', '')
        label = key_data.get('label', '')
        x = key_data.get('x', 0)
        y = key_data.get('y', 0)
        
        target_layer = _parse_layer_from_behavior(label, behavior, parameter)
        if target_layer is not None and target_layer != 0:
            is_momentary = _is_momentary_access(behavior)
            access_list.append(LayerAccess(
                target_layer=target_layer,
                source_layer=0,
                source_x=float(x),
                source_y=float(y),
                hand=_hand_from_x(x),
                is_momentary=is_momentary,
                access_key_label=label,
            ))
    
    # Step 2: Find access keys on intermediate layers (for depth-2 layers)
    layers = canonical_data.get('layers', {})
    for layer_id, layer_data in layers.items():
        if not layer_id or layer_id == '0':
            continue
        source_layer = int(layer_id)
        keys = layer_data.get('keys', {})
        
        for coord, key_data in keys.items():
            behavior = key_data.get('behavior', '')
            parameter = key_data.get('parameter', '')
            label = key_data.get('label', '')
            x = key_data.get('x', 0)
            y = key_data.get('y', 0)
            
            target_layer = _parse_layer_from_behavior(label, behavior, parameter)
            if target_layer is not None and target_layer not in (0, source_layer):
                is_momentary = _is_momentary_access(behavior)
                # Determine which hand is used for this access key
                hand = _hand_from_x(x)
                
                # For intermediate layers, if the access key is on a non-thumb
                # position, the thumb occupancy is determined by the source layer's
                # access path. But for simplicity, we use the key's hand.
                access_list.append(LayerAccess(
                    target_layer=target_layer,
                    source_layer=source_layer,
                    source_x=float(x),
                    source_y=float(y),
                    hand=hand,
                    is_momentary=is_momentary,
                    access_key_label=label,
                ))
    
    return access_list


def load_canonical(path: str) -> Tuple[List[Position], np.ndarray, List[LayerAccess]]:
    """Load canonical.json and extract positions with frozen info and layer access."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    positions = []
    frozen = []
    
    physical_grid = data.get("physical_grid", {})
    position_metas = physical_grid.get("positions", [])
    
    layers = data.get("layers", {})
    
    # Build positions across all layers
    for layer_key, layer_data in layers.items():
        if not layer_key or not layer_key.strip():
            continue
        layer_num = int(layer_key)
        layer_keys = layer_data.get("keys", [])
        
        for pos_idx, (coord, key_data) in enumerate(layer_keys.items()):
            if pos_idx >= len(position_metas):
                continue
            
            pos_meta = position_metas[pos_idx]
            x = key_data.get('x', pos_meta.get('x', 0))
            y = key_data.get('y', pos_meta.get('y', 0))
            
            # Parse finger info
            finger_map = {
                "thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4,
                "far_pinky": 4, "index_stretch": 1,
            }
            finger_str = pos_meta.get("finger", "index")
            finger = finger_map.get(finger_str, 1)
            
            # Compute effort from row type
            row_type = pos_meta.get("row_type", "middle")
            effort_map = {"top": 1.5, "upper": 1.2, "middle": 1.0, "lower": 1.3, "bottom": 2.0, "thumb": 0.8}
            effort = effort_map.get(row_type, 1.0)
            
            is_thumb = pos_meta.get("zone") == "thumb" or finger == 0
            is_frozen = layer_num == 7  # L7 is frozen
            
            # On L0, main typing area is frozen. Only Space and Return are
            # frozen thumb raw inputs; the other thumb positions stay mutable.
            if layer_num == 0 and not is_thumb and pos_meta.get("zone") == "finger":
                is_frozen = True
            if layer_num == 0 and is_thumb:
                raw_key_id = _normalize_raw_key_id(key_data.get("parameter", ""))
                if raw_key_id in L0_FROZEN_THUMB_RAW_KEYS:
                    is_frozen = True
            
            pos = Position(
                gene_idx=len(positions),
                layer=layer_num,
                x=float(x),
                y=float(y),
                hand=pos_meta.get("hand", "left"),
                finger=finger,
                effort=effort,
                is_thumb=is_thumb,
                is_frozen=is_frozen,
                row=0,
                col=pos_idx,
            )
            positions.append(pos)
            frozen.append(is_frozen)
    
    frozen_mask = np.array(frozen, dtype=bool)
    
    # Build layer access mapping
    layer_access = _build_layer_access(data)
    
    return positions, frozen_mask, layer_access


def load_shortcuts(path: str, canonical_data: Optional[dict] = None) -> List[Shortcut]:
    """Load app_shortcut_scores.json and build Shortcut objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    shortcuts = []
    seen_keys = set()
    permanent_l0_raw = _permanent_l0_raw_keys(canonical_data or {})

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
    
    for app_data in data.get("apps", []):
        app_name = app_data.get("name", "unknown")
        for sc_data in app_data.get("shortcuts", []):
            keys = sc_data.get("keys", "")
            if keys in seen_keys:
                continue
            seen_keys.add(keys)
            
            # Extract base_key from keys if not provided
            base_key = sc_data.get("base_key", "")
            if not base_key:
                parts = keys.replace('+', ' ').split()
                base_key = parts[-1] if parts else keys
            
            # Extract modifiers from keys
            modifiers = []
            for part in keys.replace('+', ' ').split()[:-1]:
                modifiers.append(part)

            raw_key_id = _normalize_raw_key_id(base_key) if not modifiers and "+" not in keys else None
            if raw_key_id is not None and raw_key_id in permanent_l0_raw:
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
    
    return shortcuts


def load_usage_stats(path: str) -> Optional[UsageData]:
    """Load usage_stats.json if it exists."""
    if not os.path.exists(path):
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return UsageData(
        sequences=data.get("sequences", {}),
        chains=data.get("chains", {}),
        workflows=data.get("workflows", {}),
        by_layer_shortcut=data.get("by_layer_shortcut", {}),
        by_app=data.get("by_app", {}),
        app_time_seconds=data.get("app_time_seconds", {}),
        total_events=data.get("total_events", 0),
        blind_spots=data.get("blind_spots", {}),
    )


def build_scratch_genome(canonical_data: dict, positions: List[Position], shortcuts: List[Shortcut]) -> np.ndarray:
    """Build a pre-seeded genome from canonical layout + greedy importance fill.
    
    Ported from v1 evolve/representation.py build_scratch_genome()."""
    n_pos = len(positions)
    genome = np.full(n_pos, -1, dtype=np.int32)
    
    # Build lookups
    keys_to_sid = {}
    base_keys_to_sid = {}
    for s in shortcuts:
        if s.keys.startswith("_base_"):
            base_keys_to_sid[s.keys] = s.sid
        else:
            keys_to_sid[s.keys.upper()] = s.sid
    
    # Access keys that should always be preserved
    access_key_patterns = {
        "_base_coach_l1_hold", "_base_coach_l2_hold",
        "_base_coach_l3_hold", "_base_coach_l4_hold",
        "_base_coach_mouse_lock", "_base_coach_game_lock",
        "_base_coach_travel_toggle", "_base_coach_travel_off",
        "_base_coach_base", "_base_coach_recover_base",
    }
    
    # Step 1: Encode current canonical assignments
    layers = canonical_data.get("layers", {})
    for pos in positions:
        layer_data = layers.get(str(pos.layer), {})
        binding = layer_data.get("keys", {}).get(f"{int(pos.x)}:{int(pos.y)}", {})
        behavior = binding.get("behavior", "")
        param = binding.get("parameter", "")
        modifiers = binding.get("modifiers", [])
        
        # Skip transparent/empty slots
        if behavior.lower() in ("transparent", "none", ""):
            continue
        
        # Skip structural layer controls (but not coach keys — those are movable)
        if behavior in ("Momentary Layer", "Toggle Layer", "To Layer"):
            if not param.lower().startswith("layer::") and not param.isdigit():
                continue
        
        # Try base key lookup first
        base_key_id = _extract_base_key_id(param, behavior, modifiers)
        if base_key_id is not None:
            label = f"_base_{base_key_id}"
            sid = base_keys_to_sid.get(label)
            if sid is not None:
                genome[pos.gene_idx] = sid
                continue
        
        # Try standard shortcut lookup
        mod_names = []
        for m in modifiers:
            ml = m.lower()
            if "gui" in ml:
                mod_names.append("Win")
            elif "ctrl" in ml:
                mod_names.append("Ctrl")
            elif "shift" in ml:
                mod_names.append("Shift")
            elif "alt" in ml:
                mod_names.append("Alt")
        
        # Sort: Win, Ctrl, Shift, Alt
        MOD_ORDER = {"Win": 0, "Ctrl": 1, "Shift": 2, "Alt": 3}
        mod_names.sort(key=lambda m: MOD_ORDER.get(m, 9))
        
        # Extract base key from parameter
        base = param.upper().replace("KEYBOARD ", "").split(" AND ")[0].split(" ")[0]
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if param.upper() == f"KEYBOARD {letter}":
                base = letter
                break
        
        shortcut_key = "+".join(mod_names + [base]) if mod_names else base
        sid = keys_to_sid.get(shortcut_key.upper())
        if sid is not None:
            genome[pos.gene_idx] = sid
    
    # Step 2: Preserve access keys on non-L0 layers (structural keys)
    for i, pos in enumerate(positions):
        if genome[i] >= 0:
            sid = int(genome[i])
            skey = shortcuts[sid].keys
            if skey in access_key_patterns or skey.startswith(("_base_toggle_layer_", "_base_momentary_layer_", "_base_to_layer_")):
                # Keep access keys on their layer
                pass
    
    # Step 3: Greedy fill remaining positions with high-importance shortcuts
    assigned_sids = set(int(g) for g in genome if g >= 0)
    unplaced = [s for s in shortcuts if s.sid not in assigned_sids and s.importance >= 1.0]
    unplaced.sort(key=lambda s: -s.importance)
    
    # Empty mutable positions: all non-L0 + open L0 thumbs
    empty_positions = [(i, positions[i]) for i in range(n_pos)
                       if genome[i] < 0 and not positions[i].is_frozen]
    empty_positions.sort(key=lambda x: x[1].effort)
    
    placed = set()
    for s in unplaced:
        if s.sid in placed:
            continue
        for idx, pos in empty_positions:
            if genome[idx] < 0:
                genome[idx] = s.sid
                placed.add(s.sid)
                break
    
    return genome


def _extract_base_key_id(param: str, behavior: str, modifiers: List[str]) -> Optional[str]:
    """Extract base key identifier for _base_* shortcut lookup."""
    p = param.lower()
    b = behavior.lower()
    
    # Coach behaviors
    coach_keys = {
        "coach_l1_hold", "coach_l2_hold", "coach_l3_hold", "coach_l4_hold",
        "coach_mouse_lock", "coach_game_lock", "coach_base",
        "coach_travel_toggle", "coach_travel_off", "coach_recover_base",
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


def build_layout(build_dir: str, config: dict = None) -> Layout:
    """Build a Layout from v1 data files.
    
    Optional config dict can contain 'shortcut_importance_overrides' to
    adjust shortcut importance without editing the source data.
    """
    canonical_path = os.path.join(build_dir, "canonical.json")
    with open(canonical_path, "r", encoding="utf-8") as f:
        canonical_data = json.load(f)

    positions, frozen_mask, layer_access = load_canonical(canonical_path)
    shortcuts = load_shortcuts(os.path.join(build_dir, "app_shortcut_scores.json"), canonical_data)
    
    # Apply importance overrides from config
    if config:
        overrides = config.get("shortcut_importance_overrides", {})
        if overrides:
            updated = []
            for sc in shortcuts:
                if sc.keys in overrides:
                    updated.append(Shortcut(
                        sid=sc.sid,
                        keys=sc.keys,
                        action=sc.action,
                        app=sc.app,
                        importance=float(overrides[sc.keys]),
                        category=sc.category,
                        modifiers=sc.modifiers,
                        base_key=sc.base_key,
                        is_capability=sc.is_capability,
                        is_l0_only=sc.is_l0_only,
                        complexity=sc.complexity,
                        preferred_hand=sc.preferred_hand,
                        primary_app=sc.primary_app,
                        app_demand=sc.app_demand,
                    ))
                else:
                    updated.append(sc)
            shortcuts = updated
    
    usage = load_usage_stats(os.path.join(build_dir, "usage_stats.json"))
    
    # Build layer_to_indices mapping
    layer_to_indices = {}
    for layer in set(p.layer for p in positions):
        indices = [p.gene_idx for p in positions if p.layer == layer]
        layer_to_indices[layer] = np.array(indices, dtype=np.int32)
    
    # Build pre-seeded genome from canonical + greedy fill
    genome = build_scratch_genome(canonical_data, positions, shortcuts)
    
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
    sid_lookup = {s.keys: s.sid for s in shortcuts}
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
