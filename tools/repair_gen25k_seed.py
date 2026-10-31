#!/usr/bin/env python3
"""Rebuild the verified gen25k layout under the versioned L1 owner contract."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolution.acceptance import (
    _momentary_hold_target_position_report,
    _required_relative_layout_report,
    build_acceptance_report,
)
from evolution.completion_cluster import analyze_completion_cluster
from evolution.layout_contract import CONTRACT, CONTRACT_SHA256, audit_layout_contract
from run_evolution import analyze_duplicates
from tools._common import checkpoint_to_layout, load_layout


TOOLS = ROOT.parent / "charybdis-tools"
ZMK = ROOT.parent / "charybdis-zmk-config"
DEFAULT_CHECKPOINT = ROOT / "build/runs/validated_30k_20261026T132331Z/v2_checkpoint_gen25000.json"
DEFAULT_EXPORT = ZMK / "layout/final_user_layout_v2.json"
DEFAULT_CSV = ZMK / "layout/keybindings_explained.csv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SeedRepair:
    def __init__(self, layout):
        self.layout = layout
        self.genome = np.asarray(layout.genome, dtype=np.int32).copy()
        self.positions = layout.positions
        self.by_coord = {
            (int(p.layer), int(p.x), int(p.y)): int(p.gene_idx)
            for p in self.positions
        }
        self.sid_by_key = {str(s.keys): int(s.sid) for s in layout.shortcuts}
        self.shortcut_by_sid = {int(s.sid): s for s in layout.shortcuts}
        self.moves = []
        self.protected_sids = set()
        for key in CONTRACT["required_l1_shortcuts"]:
            if key in self.sid_by_key:
                self.protected_sids.add(self.sid_by_key[key])
        for key in ("MB1", "MB2", "MB3", "MB4", "MB5"):
            if key in self.sid_by_key:
                self.protected_sids.add(self.sid_by_key[key])
        for cluster in getattr(layout, "semantic_clusters", ()):
            self.protected_sids.update(int(member.get("sid", -1))
                                       for member in cluster.get("members", ()))

    def sid(self, key):
        if key not in self.sid_by_key:
            raise KeyError(f"Shortcut absent from corpus: {key}")
        return self.sid_by_key[key]

    def index(self, layer, x, y):
        return self.by_coord[(int(layer), int(x), int(y))]

    def record(self, idx, before, after, reason):
        if before == after:
            return
        p = self.positions[idx]
        self.moves.append({
            "idx": int(idx), "layer": int(p.layer), "x": float(p.x), "y": float(p.y),
            "from": self.shortcut_by_sid[before].keys if before >= 0 else "Transparent",
            "to": self.shortcut_by_sid[after].keys if after >= 0 else "Transparent",
            "reason": reason,
        })

    def set(self, idx, sid, reason):
        before = int(self.genome[idx])
        sid = int(sid)
        self.genome[idx] = sid
        self.record(idx, before, sid, reason)

    def swap(self, a, b, reason):
        a, b = int(a), int(b)
        if a == b:
            return
        left, right = int(self.genome[a]), int(self.genome[b])
        self.genome[a], self.genome[b] = right, left
        self.record(a, left, right, reason)
        self.record(b, right, left, reason)

    def swap_layers(self, layer_a, layer_b):
        coords_a = {(int(p.x), int(p.y)): int(p.gene_idx) for p in self.positions if int(p.layer) == layer_a}
        coords_b = {(int(p.x), int(p.y)): int(p.gene_idx) for p in self.positions if int(p.layer) == layer_b}
        if set(coords_a) != set(coords_b):
            raise RuntimeError("L1/L5 physical coordinate sets differ")
        for coord in sorted(coords_a):
            self.swap(coords_a[coord], coords_b[coord], "move complete prior L5 mouse workflow to fixed L1; preserve displaced L1 bindings on L5")

    def move_group(self, keys, targets, reason):
        sids = [self.sid(key) for key in keys]
        target_indices = [self.index(*target) for target in targets]
        if len(set(target_indices)) != len(target_indices) or len(set(sids)) != len(sids):
            raise ValueError("Group targets and shortcut IDs must be unique")
        source_indices = []
        for sid in sids:
            occurrences = np.flatnonzero(self.genome == sid).tolist()
            if not occurrences:
                raise RuntimeError(f"Required group member absent: {self.shortcut_by_sid[sid].keys}")
            source_indices.append(min(
                occurrences,
                key=lambda i: (0 if int(self.positions[i].layer) in (1, 5) else 1,
                               0 if int(self.positions[i].layer) == 1 else 1, i),
            ))
        old = [int(self.genome[i]) for i in target_indices]
        target_set = set(target_indices)
        vacated = [i for i in source_indices if i not in target_set]
        self.genome[target_indices] = np.asarray(sids, dtype=np.int32)
        displaced = [sid for sid in old if sid not in sids and sid >= 0]
        for sid, idx in zip(displaced, vacated):
            before = int(self.genome[idx])
            self.genome[idx] = sid
            self.record(idx, before, sid, reason + "; relocate displaced binding")
        for idx, sid in zip(target_indices, sids):
            before = old[target_indices.index(idx)]
            self.record(idx, before, sid, reason)
        if len(displaced) > len(vacated):
            for sid in displaced[len(vacated):]:
                empties = [i for i, value in enumerate(self.genome) if value < 0 and not self.positions[i].is_frozen]
                if not empties:
                    raise RuntimeError(f"No slot to preserve displaced shortcut {sid}")
                idx = empties[0]
                self.set(idx, sid, reason + "; preserve displaced binding in empty slot")

    def move_sid(self, sid, target, reason, *, displaced_target=None,
                 expected_displaced_sid=None):
        sid = int(sid)
        target_idx = self.index(*target)
        if int(self.genome[target_idx]) == sid:
            return
        occurrences = np.flatnonzero(self.genome == sid).tolist()
        if not occurrences:
            raise RuntimeError(f"Cannot move absent shortcut {self.shortcut_by_sid[sid].keys}")
        source_idx = min(occurrences, key=lambda i: (
            0 if int(self.positions[i].layer) in (1, 5) else 1,
            0 if int(self.positions[i].layer) == 1 else 1,
            i,
        ))
        displaced_sid = int(self.genome[target_idx])
        if displaced_target is not None and displaced_sid >= 0:
            if expected_displaced_sid is not None and displaced_sid != int(expected_displaced_sid):
                raise RuntimeError(
                    "Unexpected binding at coordinated move target: "
                    f"{self.shortcut_by_sid[displaced_sid].keys}"
                )
            displaced_idx = self.index(*displaced_target)
            if displaced_idx in (source_idx, target_idx):
                raise RuntimeError("Displaced-binding destination overlaps move source or target")
            old_destination_sid = int(self.genome[displaced_idx])
            # Three-way relocation: put the requested binding at target, move
            # its displaced binding to its required group position, and carry
            # that position's old contents back to the vacated source. This
            # prevents an incidental cross-layer duplicate at the source.
            self.set(target_idx, sid, reason)
            self.set(displaced_idx, displaced_sid,
                     reason + "; preserve displaced binding at required group position")
            self.set(source_idx, old_destination_sid,
                     reason + "; preserve displaced group-position contents")
            return
        self.swap(source_idx, target_idx, reason)

    def move_key(self, key, target, reason):
        self.move_sid(self.sid(key), target, reason)

    def repair_thumb_violations(self):
        for _ in range(len(self.positions) * 2):
            report = audit_layout_contract(self.layout.clone_with(genome=self.genome))
            rows = report["details"]["momentary_thumb_violations"]
            if not rows:
                return
            row = rows[0]
            src = int(row["idx"])
            layer = int(row["layer"])
            # Prefer an empty non-thumb position on the same layer so access
            # reachability and layer workflow membership remain unchanged.
            candidates = [
                int(p.gene_idx) for p in self.positions
                if int(p.layer) == layer and not p.is_thumb and self.genome[p.gene_idx] < 0
            ]
            if candidates:
                self.swap(src, candidates[0], "clear occupied thumb held along reachable momentary access path")
                continue
            # If the layer is full, move a non-group workflow action to a free
            # non-thumb position elsewhere, then move the blocked key into its slot.
            free = [
                int(p.gene_idx) for p in self.positions
                if int(p.layer) == layer and not p.is_thumb and self.genome[p.gene_idx] >= 0
                and int(self.genome[p.gene_idx]) not in self.protected_sids
                and not self.shortcut_by_sid[int(self.genome[p.gene_idx])].is_layer_access
            ]
            if not free:
                raise RuntimeError(f"Cannot safely clear thumb violation: {row}")
            target = free[0]
            empty_elsewhere = [
                int(p.gene_idx) for p in self.positions
                if not p.is_frozen and not p.is_thumb and self.genome[p.gene_idx] < 0
            ]
            if not empty_elsewhere:
                raise RuntimeError("No free non-thumb slot to preserve bindings during thumb clearance")
            displaced = int(self.genome[target])
            empty = empty_elsewhere[0]
            self.set(empty, displaced, "preserve displaced workflow binding during thumb clearance")
            self.set(target, int(self.genome[src]), "move held thumb binding to non-thumb position")
            self.set(src, -1, "clear occupied thumb held along reachable momentary access path")
        raise RuntimeError("Thumb clearance repair exceeded 160 moves")

    def repair_hold_target_conflicts(self):
        for _ in range(len(self.positions) * 2):
            current_layout = self.layout.clone_with(genome=self.genome)
            report = _momentary_hold_target_position_report(current_layout)
            conflicts = report["conflicts"]
            if not conflicts:
                return
            target_idx = int(conflicts[0]["target_idx"])
            pos = self.positions[target_idx]
            blocked = set()
            for idx, source_pos in enumerate(self.positions):
                sid = int(self.genome[idx])
                if sid < 0:
                    continue
                shortcut = self.shortcut_by_sid[sid]
                if (shortcut.is_layer_access and shortcut.access_is_momentary
                        and not str(shortcut.keys).startswith("@scroll:")
                        and int(shortcut.access_target_layer) != int(source_pos.layer)):
                    blocked.add((int(shortcut.access_target_layer),
                                 float(source_pos.x), float(source_pos.y)))
            hold_idx = int(conflicts[0]["hold_idx"])
            hold_pos = self.positions[hold_idx]
            hold = self.shortcut_by_sid[int(self.genome[hold_idx])]
            fixed_essential = (int(hold_pos.layer) == 0 and
                               int(hold.access_target_layer) == 1 and
                               float(hold_pos.x) == 3.0 and float(hold_pos.y) == 4.0)
            if not fixed_essential and hold.is_layer_access:
                reachable = {0}
                edges = []
                for i, source in enumerate(self.positions):
                    sid = int(self.genome[i])
                    if sid < 0 or i == hold_idx:
                        continue
                    shortcut = self.shortcut_by_sid[sid]
                    if (shortcut.is_layer_access and shortcut.access_is_momentary
                            and not str(shortcut.keys).startswith("@scroll:")
                            and int(shortcut.access_target_layer) != int(source.layer)):
                        edges.append((int(source.layer), int(shortcut.access_target_layer)))
                for _ in range(32):
                    newer = reachable | {b for a, b in edges if a in reachable}
                    if newer == reachable:
                        break
                    reachable = newer
                if int(hold.access_target_layer) in reachable:
                    self.set(hold_idx, -1,
                             "remove redundant ordinary access edge; target retains another hold path")
                    continue
            alternate = [] if fixed_essential else [
                int(p.gene_idx) for p in self.positions
                if int(p.layer) == int(hold_pos.layer) and self.genome[p.gene_idx] < 0
                and not p.is_thumb and not p.is_frozen
                and (int(hold.access_target_layer), float(p.x), float(p.y)) not in blocked
                and int(self.genome[self.index(hold.access_target_layer, p.x, p.y)]) < 0
            ]
            if alternate:
                self.swap(hold_idx, alternate[0],
                          "move ordinary hold to an unoccupied corresponding target position")
                continue
            empty = [
                int(p.gene_idx) for p in self.positions
                if int(p.layer) == int(pos.layer) and p.gene_idx != target_idx
                and self.genome[p.gene_idx] < 0 and not p.is_thumb and not p.is_frozen
                and (int(p.layer), float(p.x), float(p.y)) not in blocked
            ]
            if not empty:
                raise RuntimeError(f"No same-layer slot to clear hold target: {conflicts[0]}")
            self.swap(target_idx, empty[0], "clear key hidden beneath a reachable ordinary momentary hold")
        raise RuntimeError(f"Hold-target repair exceeded move limit; remaining conflicts: {conflicts}")

    def remove_same_layer_duplicates(self):
        for layer in range(11):
            if layer == 7:
                continue
            seen = set()
            for pos in self.positions:
                if int(pos.layer) != layer:
                    continue
                idx = int(pos.gene_idx)
                sid = int(self.genome[idx])
                if sid < 0:
                    continue
                key = str(self.shortcut_by_sid[sid].keys)
                if key in seen:
                    self.set(idx, -1, "remove redundant same-layer duplicate; keep one reachable binding")
                else:
                    seen.add(key)

    def remove_unsupported_cross_layer_duplicates(self, duplicate_report):
        """Keep the best evidenced placement for each unsupported duplicate."""
        layer_counts = self.layout.usage_data.by_layer_shortcut
        for row in duplicate_report.get("unsupported_duplicates", ()):
            sid = int(row["sid"])
            placements = [
                (int(item["idx"]), int(item["layer"]))
                for item in row.get("positions", ())
                if int(item["idx"]) < len(self.genome)
                and int(self.genome[int(item["idx"])]) == sid
            ]
            if len(placements) < 2:
                continue
            key = str(self.shortcut_by_sid[sid].keys)
            observed = layer_counts.get(key, {}) if isinstance(layer_counts, dict) else {}
            # Prefer the layer with actual binding-specific use. Ties keep the
            # shallower layer, then lower-effort position for a deterministic
            # choice when logger data has not distinguished the placements.
            keep_idx, keep_layer = max(
                placements,
                key=lambda item: (
                    int(observed.get(str(item[1]), 0)),
                    -item[1],
                    -float(self.positions[item[0]].effort),
                    -item[0],
                ),
            )
            for idx, layer in placements:
                if idx == keep_idx:
                    continue
                count = int(observed.get(str(layer), 0))
                keep_count = int(observed.get(str(keep_layer), 0))
                self.set(
                    idx, -1,
                    f"remove unsupported duplicate; keep L{keep_layer} placement "
                    f"(layer use {keep_count} vs {count})",
                )


def repair(checkpoint_path: Path, output_dir: Path, data_dir: Path):
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    layout = checkpoint_to_layout(checkpoint, load_layout(data_dir=str(data_dir)))
    repairer = SeedRepair(layout)
    repairer.swap_layers(1, 5)

    # Rewrite all access paths that referred to the former mouse layer.
    for idx, sid_value in enumerate(repairer.genome.copy()):
        sid = int(sid_value)
        if sid < 0:
            continue
        shortcut = repairer.shortcut_by_sid[sid]
        key = str(shortcut.keys)
        if key.startswith("@access:L5:"):
            suffix = key.rsplit(":", 1)[-1]
            target_sid = repairer.sid(f"@access:L1:{suffix}")
            repairer.set(idx, target_sid, "rewrite former mouse-layer access target L5 to fixed L1")
        elif key == "@scroll:L5:hold":
            repairer.set(idx, repairer.sid("@scroll:L1:hold"), "rewrite former mouse-layer scroll target L5 to fixed L1")

    # Retain two movable current-layer scroll modes on L2 and L4. Their old
    # source positions are exchanged with their new layer-local positions.
    repairer.move_sid(
        repairer.sid("@scroll:L2:hold"), (2, 10, 2),
        "place additional current-layer Scroll on L2",
        displaced_target=(8, 2, 1),
        expected_displaced_sid=repairer.sid("Ctrl+6"),
    )
    repairer.move_sid(repairer.sid("@scroll:L4:hold"), (4, 10, 2), "place additional current-layer Scroll on L4")

    # Convert all remaining non-contract Scroll holds into ordinary access
    # holds where they provided a distinct target, and remove useless self holds.
    for idx, sid_value in enumerate(repairer.genome.copy()):
        sid = int(sid_value)
        if sid < 0:
            continue
        shortcut = repairer.shortcut_by_sid[sid]
        if not str(shortcut.keys).startswith("@scroll:"):
            continue
        pos = repairer.positions[idx]
        layer = int(pos.layer)
        if (sid == repairer.sid("@scroll:L1:hold") and layer == 1
                and float(pos.x) == 10.0 and float(pos.y) == 2.0):
            continue
        if (sid == repairer.sid("@scroll:L2:hold") and layer == 2
                and float(pos.x) == 10.0 and float(pos.y) == 2.0):
            continue
        if (sid == repairer.sid("@scroll:L4:hold") and layer == 4
                and float(pos.x) == 10.0 and float(pos.y) == 2.0):
            continue
        target = int(shortcut.access_target_layer)
        if 0 < target < 11 and target != 7 and target != layer:
            ordinary_sid = repairer.sid(f"@access:L{target}:hold")
            duplicate = any(
                int(repairer.positions[j].layer) == layer and int(repairer.genome[j]) == ordinary_sid
                for j in range(len(repairer.genome)) if j != idx
            )
            repairer.set(idx, -1 if duplicate else ordinary_sid,
                         "replace non-contract Scroll edge with ordinary layer hold")
        else:
            repairer.set(idx, -1, "remove extra/self Scroll; retain only the three contracted modes")

    # The authoritative Studio exporter deliberately omits these corpus items.
    # Keep the seed/export comparison exact by recording and removing them.
    for idx, sid_value in enumerate(repairer.genome.copy()):
        sid = int(sid_value)
        if sid < 0 or repairer.positions[idx].is_frozen:
            continue
        shortcut = repairer.shortcut_by_sid[sid]
        if shortcut.base_key == "Pause" or str(shortcut.keys) in {"Ctrl+K S", "gg", "gi", "yy"}:
            repairer.set(idx, -1, "remove binding omitted by the ZMK Studio exporter")

    # Fixed L0 entry and an alternate direct ordinary path to L2.
    repairer.set(repairer.index(0, 3, 4), repairer.sid("@access:L1:hold"), "fixed essential L0 x3,y4 ordinary hold to L1")
    repairer.set(repairer.index(0, 5, 5), -1, "remove redundant second L0-to-L1 hold; fixed x3,y4 hold remains essential")

    # Required L1 shortcuts and sequence group, all outside the top row.
    repairer.move_group(["Ctrl+X", "Ctrl+C", "Ctrl+V"],
                        [(1, 1, 3), (1, 2, 3), (1, 3, 3)],
                        "preserve ordered clipboard sequence group on low-effort L1 positions")
    repairer.move_group(["Win+Shift+Left", "Win+Shift+Right"],
                        [(1, 1, 2), (1, 2, 2)],
                        "place adjacent monitor-move pair on L1")
    repairer.move_group(["Ctrl+-", "Ctrl++"],
                        [(1, 3, 2), (1, 4, 2)],
                        "place adjacent zoom pair on L1")
    repairer.move_group(["Win+Left", "Win+Right"],
                        [(1, 10, 1), (1, 11, 1)],
                        "place adjacent window-snap pair on L1")
    for key, target in [
        ("Win+Tab", (1, 1, 1)),
        ("Win+Alt+Space", (1, 2, 1)),
        ("Ctrl+S", (1, 3, 1)),
        ("Ctrl+A", (1, 5, 3)),
        ("Win+Shift+S", (1, 4, 3)),
    ]:
        repairer.move_key(key, target, "place mandatory shortcut on legal low-effort L1 position")

    # Keep the dedicated Walk This Way trigger directly beside normal CmdPal.
    # Move Ctrl+S one key right as part of the same coordinated arrangement.
    repairer.move_group(
        ["Win+Alt+Space", "Ctrl+Alt+Win+F11"],
        [(1, 2, 1), (1, 3, 1)],
        "group Walk This Way beside normal PowerToys Command Palette on L1",
    )
    repairer.move_key(
        "Ctrl+S", (1, 4, 1),
        "preserve required low-effort Ctrl+S beside the grouped Command Palette keys",
    )

    # Restore established groups as atomic arrangements after relocating the
    # owner-mandated L1 shortcuts.
    repairer.move_group(["=", "-", "`", "]", "<"],
                        [(10, 10, 0), (10, 9, 0), (10, 8, 0),
                         (10, 8, 1), (10, 8, 3)],
                        "restore exact Norwegian raw-key grouping on L10")
    repairer.move_group([f"Ctrl+{n}" for n in range(1, 10)],
                        [(8, x, y) for y in range(3) for x in range(3)],
                        "restore Ctrl+1 through Ctrl+9 3x3 shortcut group")
    repairer.move_group(["Ctrl+Left", "Ctrl+Right"],
                        [(8, 0, 3), (8, 1, 3)],
                        "restore adjacent Ctrl+Left and Ctrl+Right group")
    repairer.move_group(["Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z"],
                        [(8, 7, 3), (8, 8, 3), (8, 9, 3)],
                        "restore ordered undo/redo history group")
    repairer.move_group(["Ctrl+F6", "Ctrl+Shift+F6"],
                        [(8, 2, 3), (8, 3, 3)],
                        "restore adjacent section navigation group")
    repairer.move_group(["Ctrl+Home", "Ctrl+End"],
                        [(8, 4, 3), (8, 5, 3)],
                        "restore adjacent document start/end group")

    # Complete ordinary-hold reachability after coordinated group moves free
    # positions in the displaced former mouse layer.
    for target in (5, 8):
        candidates = [
            int(p.gene_idx) for p in repairer.positions
            if int(p.layer) == 6 and not p.is_thumb and repairer.genome[p.gene_idx] < 0
            and int(repairer.genome[repairer.index(target, p.x, p.y)]) < 0
        ]
        if not candidates:
            held = set()
            for i, source in enumerate(repairer.positions):
                sid = int(repairer.genome[i])
                if sid >= 0:
                    shortcut = repairer.shortcut_by_sid[sid]
                    if (shortcut.is_layer_access and shortcut.access_is_momentary
                            and not str(shortcut.keys).startswith("@scroll:")):
                        held.add((int(shortcut.access_target_layer), float(source.x), float(source.y)))
            freeing = []
            destinations = [int(p.gene_idx) for p in repairer.positions
                            if int(p.layer) == 6 and not p.is_thumb
                            and repairer.genome[p.gene_idx] < 0
                            and (6, float(p.x), float(p.y)) not in held]
            for source in repairer.positions:
                if int(source.layer) != target:
                    continue
                source_idx = int(source.gene_idx)
                sid = int(repairer.genome[source_idx])
                if (sid < 0 or sid in repairer.protected_sids
                        or repairer.shortcut_by_sid[sid].is_layer_access):
                    continue
                target_slot = repairer.index(6, source.x, source.y)
                if (repairer.genome[target_slot] < 0 and
                        (6, float(source.x), float(source.y)) not in held and destinations):
                    freeing.append((source_idx, target_slot, destinations[0], sid))
                    break
            if not freeing:
                raise RuntimeError(f"No safe movable binding to free a target slot on L{target}")
            old_idx, access_idx, destination_idx, displaced_sid = freeing[0]
            repairer.set(destination_idx, displaced_sid,
                         f"preserve displaced L{target} binding while creating ordinary access path")
            repairer.set(old_idx, -1, f"free L{target} access coordinate for ordinary hold")
            candidates = [access_idx]
        idx = candidates[0]
        repairer.set(idx, repairer.sid(f"@access:L{target}:hold"),
                     f"restore ordinary momentary access path from reachable L6 to generated L{target}")

    # Put L1's lock on the right thumb while retaining its return key on the
    # left side held by the essential L0 access key.
    lock_sid = repairer.sid("@access:L1:toggle")
    return_sid = repairer.sid("@access:L0:return")
    lock_positions = [i for i, sid in enumerate(repairer.genome) if int(sid) == lock_sid and int(repairer.positions[i].layer) == 1]
    return_positions = [i for i, sid in enumerate(repairer.genome) if int(sid) == return_sid and int(repairer.positions[i].layer) == 1]
    target_lock = repairer.index(1, 7, 5)
    target_return = repairer.index(1, 5, 5)
    if lock_positions and return_positions and target_return in lock_positions and target_lock in return_positions:
        repairer.swap(target_return, target_lock, "place reachable L1 self-lock on free right thumb and return-to-L0 on restricted left thumb")

    for _ in range(4):
        repairer.repair_thumb_violations()
        repairer.repair_hold_target_conflicts()
        repairer.remove_same_layer_duplicates()

    duplicate_report = analyze_duplicates(layout.clone_with(genome=repairer.genome))
    repairer.remove_unsupported_cross_layer_duplicates(duplicate_report)

    repaired_layout = layout.clone_with(genome=repairer.genome)
    owner_audit = audit_layout_contract(repaired_layout)
    duplicate_report = analyze_duplicates(repaired_layout)
    completion_report = analyze_completion_cluster(repaired_layout)
    acceptance = build_acceptance_report(
        repaired_layout,
        duplicate_report=duplicate_report,
        completion_cluster_report=completion_report,
    )
    relative = _required_relative_layout_report(repaired_layout)
    output_dir.mkdir(parents=True, exist_ok=True)
    genome_path = output_dir / "gen25k_corrected_seed.json"
    genome_data = {
        "genome": repairer.genome.astype(int).tolist(),
        "generation": 0,
        "source_generation": 25000,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_genome_sha256": sha256(json.dumps(checkpoint["best_genome"], separators=(",", ":")).encode()),
        "contract_version": CONTRACT["version"],
        "contract_sha256": CONTRACT_SHA256,
        "usage_stats_sha256": sha256((data_dir / "usage_stats.json").read_bytes()),
        "accepted": bool(owner_audit["acceptance_pass"]),
    }
    genome_path.write_text(json.dumps(genome_data, indent=2) + "\n", encoding="utf-8")
    export_path = DEFAULT_EXPORT
    csv_path = DEFAULT_CSV
    report = {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256(checkpoint_bytes),
        "source_generation": int(checkpoint.get("generation", -1)),
        "source_best_source": checkpoint.get("best_source"),
        "source_best_generation": checkpoint.get("best_generation"),
        "source_genome_sha256": genome_data["source_genome_sha256"],
        "previously_verified_export": {
            "path": str(export_path.resolve()),
            "sha256": sha256(export_path.read_bytes()) if export_path.is_file() else None,
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256(csv_path.read_bytes()) if csv_path.is_file() else None,
        },
        "contract_version": CONTRACT["version"],
        "contract_sha256": CONTRACT_SHA256,
        "usage_stats_sha256": genome_data["usage_stats_sha256"],
        "corrected_genome_path": str(genome_path.resolve()),
        "corrected_genome_sha256": sha256(genome_path.read_bytes()),
        "owner_contract": owner_audit,
        "optimizer_acceptance": acceptance,
        "duplicate_report": duplicate_report,
        "completion_cluster_report": completion_report,
        "required_relative_layouts": relative,
        "moves": repairer.moves,
    }
    report_path = output_dir / "gen25k_seed_correction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "corrected_genome": str(genome_path),
        "correction_report": str(report_path),
        "owner_contract_pass": owner_audit["acceptance_pass"],
        "owner_contract_failures": owner_audit["failed_checks"],
        "optimizer_acceptance_pass": acceptance["optimizer_side_pass"],
        "optimizer_failures": [k for k, v in acceptance["optimizer_side_checks"].items() if not v],
        "relative_failures": [row["name"] for row in relative["failures"]],
        "unsupported_duplicate_count": len(duplicate_report.get("unsupported_duplicates", [])),
        "moves": len(repairer.moves),
    }, indent=2))
    return owner_audit["acceptance_pass"] and acceptance["optimizer_side_pass"] and relative["acceptance_pass"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Layout and usage snapshot used for the acceptance audit")
    args = parser.parse_args()
    return 0 if repair(args.checkpoint.resolve(), args.output_dir.resolve(), args.data_dir.resolve()) else 4


if __name__ == "__main__":
    raise SystemExit(main())
