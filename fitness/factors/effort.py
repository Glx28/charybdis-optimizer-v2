"""Effort factor: penalizes high-effort positions and nested layer access for important shortcuts.

NOTE: EffortFactor.compute() is a Python-only legacy path used in unit tests.
Production scoring uses the compiled kernel (fitness/kernel.py _single_genome),
which builds layer-access costs dynamically from the evolved genome, never from
the legacy static layout.layer_access field.
"""
import numpy as np
from core import Layout
from fitness import FitnessFactor


class EffortFactor(FitnessFactor):
    """Sum of importance * (position effort + layer access cost) for all assigned positions. Lower is better."""
    name = "effort"

    def _compute_layer_access_costs_from_genome(self, layout: Layout) -> dict:
        """Compute layer access costs from the evolved genome bindings.

        Reads shortcuts assigned in the genome that have ``is_layer_access=True``
        to build the reachability graph.  This is authoritative for evolved
        candidates because access buttons are genome capabilities, not fixed
        canonical positions.

        If no access shortcuts exist in the genome at all (e.g. a minimal test
        layout), all layers are assumed directly accessible (cost 0.0) so that
        the factor degrades gracefully instead of adding large unreachability
        penalties to every layer.
        """
        all_layers = set(p.layer for p in layout.positions)
        costs: dict = {0: 0.0}
        access_graph: dict = {}
        for i, sid in enumerate(layout.genome):
            sid = int(sid)
            if sid < 0 or sid >= len(layout.shortcuts):
                continue
            shortcut = layout.shortcuts[sid]
            if not shortcut.is_layer_access or shortcut.access_target_layer < 0:
                continue
            pos = layout.positions[i]
            target = shortcut.access_target_layer
            source = pos.layer
            access_graph.setdefault(target, []).append((source, pos.effort))

        # If no access shortcuts are present at all, assume every layer is
        # directly accessible from L0 (test/fallback path only).
        if not access_graph:
            return {layer: 0.0 for layer in all_layers}

        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for target, sources in access_graph.items():
                if target in costs:
                    continue
                for source_layer, effort in sources:
                    if source_layer in costs:
                        costs[target] = costs[source_layer] + effort
                        changed = True
                        break

        for layer in all_layers:
            if layer not in costs:
                costs[layer] = 5.0
        return costs

    def _compute_layer_access_costs(self, layout: Layout) -> dict:
        """Legacy/static fallback only. Not authoritative for evolved dynamic access.

        Reads ``layout.layer_access`` which is a legacy static fallback.  An
        evolved candidate may have placed access shortcuts
        in completely different positions.  Call
        ``_compute_layer_access_costs_from_genome`` instead for evolved layouts.
        """
        if not layout.layer_access:
            all_layers = set(p.layer for p in layout.positions)
            return {layer: 0.0 for layer in all_layers}

        access_graph: dict = {}
        for access in layout.layer_access:
            access_effort = 2.0
            for pos in layout.positions:
                if (pos.layer == access.source_layer
                        and abs(pos.x - access.source_x) < 0.5
                        and abs(pos.y - access.source_y) < 0.5):
                    access_effort = pos.effort
                    break
            access_graph.setdefault(access.target_layer, []).append(
                (access.source_layer, access_effort)
            )

        costs = {0: 0.0}
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for target, sources in access_graph.items():
                if target in costs:
                    continue
                for source_layer, effort in sources:
                    if source_layer in costs:
                        costs[target] = costs[source_layer] + effort
                        changed = True
                        break

        for layer in set(p.layer for p in layout.positions):
            if layer not in costs:
                costs[layer] = 5.0
        return costs

    def compute(self, layout: Layout, toggle_effort_multiplier: float = 2.5) -> float:
        """Python-only legacy path (used in unit tests).

        Uses dynamic genome-based access costs.  Production scoring uses the
        compiled kernel which performs an equivalent calculation inline.
        """
        access_costs = self._compute_layer_access_costs_from_genome(layout)
        total = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]
            layer = pos.layer
            access_cost = access_costs.get(layer, 0.0)
            pos_eff = pos.effort
            if shortcut.is_layer_access and not shortcut.access_is_momentary:
                pos_eff *= toggle_effort_multiplier
            total += shortcut.importance * (pos_eff + access_cost)
        return total
