"""Effort factor: penalizes high-effort positions and nested layer access for important shortcuts."""
import numpy as np
from core import Layout
from fitness import FitnessFactor

class EffortFactor(FitnessFactor):
    """Sum of importance * (position effort + layer access cost) for all assigned positions. Lower is better."""
    name = "effort"
    
    def _compute_layer_access_costs(self, layout: Layout) -> dict:
        """Compute the access cost (effort of layer buttons) to reach each layer from L0.
        
        Returns a dict mapping layer -> access cost. L0 has cost 0.0.
        For nested layers, traces back through source layers and sums the costs.
        If no layer access paths are defined, all layers are assumed directly accessible (cost 0.0).
        """
        # If no layer access data, assume all layers are directly accessible
        if not layout.layer_access:
            all_layers = set(p.layer for p in layout.positions)
            return {layer: 0.0 for layer in all_layers}
        
        # Build access graph: target_layer -> list of (source_layer, effort)
        access_graph = {}
        for access in layout.layer_access:
            if access.target_layer not in access_graph:
                access_graph[access.target_layer] = []
            # Find the position of the access key to get its effort
            access_effort = 2.0  # default fallback
            for pos in layout.positions:
                if pos.layer == access.source_layer and abs(pos.x - access.source_x) < 0.5 and abs(pos.y - access.source_y) < 0.5:
                    access_effort = pos.effort
                    break
            access_graph[access.target_layer].append((access.source_layer, access_effort))
        
        # Compute access costs via BFS from L0
        costs = {0: 0.0}
        changed = True
        max_iterations = 10
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for target, sources in access_graph.items():
                if target in costs:
                    continue
                for source_layer, source_effort in sources:
                    if source_layer in costs:
                        costs[target] = costs[source_layer] + source_effort
                        changed = True
                        break
        
        # For any unreachable layers, use a high default cost
        for layer in set(p.layer for p in layout.positions):
            if layer not in costs:
                costs[layer] = 5.0
        
        return costs
    
    def compute(self, layout: Layout) -> float:
        access_costs = self._compute_layer_access_costs(layout)
        total = 0.0
        for i, sid in enumerate(layout.genome):
            if sid < 0:
                continue
            shortcut = layout.shortcuts[sid]
            pos = layout.positions[i]
            layer = pos.layer
            access_cost = access_costs.get(layer, 0.0)
            total += shortcut.importance * (pos.effort + access_cost)
        return total
