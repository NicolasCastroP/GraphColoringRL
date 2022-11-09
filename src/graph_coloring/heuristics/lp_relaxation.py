from itertools import product

import networkx as nx
from typing import Optional

import gurobi as gb
from .base import DualBound
from .. import Coloring
from math import ceil


class GCFormulation:
    def __init__(self, graph, integrality=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph
        
        # some upper bound for \chi(G)
        degrees = sorted(dict(graph.degree()).values())
        self.ub = max(map(lambda i: min(degrees[-i - 1], i), range(len(degrees))))  # 3
        self.lb = 0
        if integrality:
            self.vtype = gb.GRB.INTEGER
        else:
            self.vtype = gb.GRB.CONTINUOUS

        self.model = self._build_model()
        
    def _build_model(self):
        m = gb.Model()
        colors = list(range(self.ub))
        nodes = self.graph.nodes
        x = m.addVars(list(product(nodes, colors)), lb=0, ub=1, vtype=self.vtype)
        y = m.addVar(lb=self.lb, ub=self.ub, obj=1)
        
        m.addConstrs(
            (y >= gb.quicksum(c * x[v, c] for c in colors) for v in nodes), name='max_color'
        )
        for vi, vj in self.graph.edges:
            m.addConstrs((x[vi, c] + x[vj, c] <= 1 for c in colors), name=f'const_{vi},{vj}')
        
        m.addConstrs(
            (gb.quicksum(x[v, c] for c in colors) == 1 for v in nodes), name='just_one_node'
        )
        
        m.update()
        return m
    
    def solve(self):
        self.model.optimize()
        
        return self.model.objVal


class LPRelaxation(DualBound):
    def _compute_reward_to_go(self, partial_coloring: Coloring = None) -> int:
        if partial_coloring is None:
            partial_coloring = Coloring(self.graph)
        m = GCFormulation(partial_coloring.complement_graph, integrality=False)
        return ceil(m.solve()) + 1