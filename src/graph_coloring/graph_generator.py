from itertools import product

from graph_coloring import Coloring


from typing import List

import networkx as nx
import numpy as np

from random import sample


DEFAULT_ARC_DENSITY=0.2


def generate_graph(
        number_of_nodes: int,
        number_of_colors: int,
        color_distribution: List[float] = None,
        arc_connection_method: str = None,
        arc_connection_kwargs: dict = None
):
    g = nx.complete_graph(number_of_colors)
    
    nodes = list(range(number_of_colors, number_of_nodes))
    colors = np.random.choice(g.nodes, p=color_distribution, size=number_of_nodes-number_of_colors)
    
    g.add_nodes_from(nodes)
    
    for n in g.nodes:
        if n < number_of_colors:
            g.nodes[n]['color'] = n
            g.nodes[n]['nodes'] = [n]
        else:
            color = colors[n - number_of_colors]
            g.nodes[n]['color'] = color
            g.nodes[color]['nodes'].append(n)

    g.number_of_colors = number_of_colors
    
    if arc_connection_kwargs is None:
        arc_connection_kwargs = dict(density=DEFAULT_ARC_DENSITY)
    
    create_arcs(g, method=arc_connection_method, **arc_connection_kwargs)
    
    return g


def create_coloring(g):
    coloring = Coloring(g)
    for n in g.nodes:
        coloring.color_node(n, g.nodes[n]['color'])
    return coloring


def create_arcs(g: nx.Graph, method=None, **kwargs):
    if method is None:
        _my_random_gen(g, density=kwargs.pop('density', DEFAULT_ARC_DENSITY))


def _my_random_gen(g: nx.Graph, density: float):
    for n in g.nodes:
        # Each node is connected with len(g) * density other nodes, of different color!
        feasible_nodes = sum(
            [g.nodes[color]['nodes'] for color in range(g.number_of_colors) if color != g.nodes[n]['color']],
            []
        )
        arcs = list(product([n], sample(feasible_nodes, int(len(feasible_nodes) * density))))
        g.add_edges_from(arcs)
    