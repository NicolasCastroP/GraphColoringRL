
from graph_coloring import *
import networkx as nx


def main():
    g = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)

    c = Coloring(g)
    
    c.color_node(0, 0)
    c.color_node(7, 0)
    

if __name__ == '__main__':
    main()