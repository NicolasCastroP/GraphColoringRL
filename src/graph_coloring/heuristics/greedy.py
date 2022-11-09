import numpy as np
from typing import Optional

from .base import BaseHeuristic
from graph_coloring.dynamics.base_class import Coloring, GraphColoringMDPSpace
from copy import copy


class GreedyColoring(BaseHeuristic):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, **kwargs)

    def run_heuristic(self, partial_coloring: Optional = None) -> Coloring:        
        coloring = self.run_heuristic_n_steps(partial_coloring)

        return coloring
    
    def run_heuristic_n_steps(self, partial_coloring: Optional = None, n: int = np.inf) -> Coloring:
        graph = self.graph
        if partial_coloring is None:
            coloring = Coloring(graph)
        else:
            coloring = copy(partial_coloring)        
        
        n = min(n, len(graph.nodes) - len(coloring.colored_nodes))
        i = 0
        while i < n:
            # Uses DS-tour to choose next node.
            v_t = GraphColoringMDPSpace.ds_tour_sequence(coloring)
            color = min(coloring.feasible_colors_for_node(v_t))
            coloring.color_node(v_t, color)
            i += 1
        
        return coloring
