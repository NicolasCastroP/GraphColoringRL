import networkx as nx
import os


def graph_from_dimacs(path, name):
	graph = nx.Graph()

	with open(os.path.join(path, name)) as f:
		l = f.readline()
		while l:
			l = l.replace('\n', '')
			if l[0] =='e':
				u, v = l.split(' ')[1:]
				graph.add_edge(u, v)
			l = f.readline()

	return graph
				