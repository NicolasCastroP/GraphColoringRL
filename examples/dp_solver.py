from graph_coloring.dynamic_programming import *
from graph_coloring.visualizer import draw_coloring
from utilities.counters import Timer


def main():
	g = nx.generators.random_graphs.erdos_renyi_graph(12, 0.2, seed=754)
	t = Timer('DP', verbose=True)
	t.start()
	coloring = solve_graph_coloring(g)
	t.stop()
	draw_coloring(g, coloring, show=True)


if __name__ == '__main__':
	main()

