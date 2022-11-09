from graph_coloring.heuristics import GreedyColoring
from graph_coloring.visualizer import draw_coloring
import networkx as nx

if __name__ == '__main__':
    graph = nx.generators.random_graphs.erdos_renyi_graph(100, 0.2, seed=754)
    greedy_c = GreedyColoring()
    
    coloring = greedy_c.run_heuristic(graph)
    
    draw_coloring(g, coloring, out_name='greedy')
