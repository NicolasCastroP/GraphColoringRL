import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pyvis.network import Network
import json
import numpy as np

_default_output_path = 'data/outputs'
_default_out_name = 'coloring'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def draw_coloring(graph: nx.Graph,
                  coloring: 'Coloring',
                  pos=None,
                  save=False,
                  output_path=_default_output_path,
                  out_name=_default_out_name,
                  legend=True,
                  ax=None,
                  show=False):
    colors = [coloring.get_color(n) for n in graph.nodes]
    if pos is None:
        pos = nx.spring_layout(graph)
    if legend:
        lines = nx.draw_networkx_edges(graph, pos, ax=ax)
        if ax is None:
            ax = lines.axes
        ax.set_xlim(
            [min(list(pos.values()), key=lambda x: x[0])[0] * 1.1,
             max(list(pos.values()), key=lambda x: x[0])[0] * 1.1])
        ax.set_ylim(
            [min(list(pos.values()), key=lambda x: x[1])[1] * 1.1,
             max(list(pos.values()), key=lambda x: x[1])[1] * 1.1])
        cmap = cm.get_cmap()
        norm = Normalize(vmin=0, vmax=len(coloring))

        for i, nodes in enumerate(coloring):
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=[cmap(norm(i))] * len(nodes),
                                   label=f'{i + 1}', ax=ax)

        ax.axis('off')
        ax.legend()

    else:
        nx.draw(graph, node_color=colors, pos=pos, ax=ax)
        if ax is not None:
            ax.set_xlim(
                [min(list(pos.values()), key=lambda x: x[0])[0], max(list(pos.values()), key=lambda x: x[0])[0]])
            ax.set_ylim(
                [min(list(pos.values()), key=lambda x: x[1])[1], max(list(pos.values()), key=lambda x: x[1])[1]])

    if save:
        plt.savefig(os.path.join(output_path, f'{out_name}{len(graph)}.pdf'))

    if show:
        plt.show()


def smooth(x_range, y_range, slope: float = 1):
    x_min, x_max = x_range
    y_min, y_max = y_range

    def f(x):
        return y_min + (np.tanh(slope * (x - x_min - (x_max - x_min) / 2)) + 1) * (y_max - y_min) / 2
    return f


class Visualizer:
    def __init__(self, graph: nx.Graph, pos=None, dynamic=False, pivis=True):
        # import matplotlib
        # matplotlib.use('TkAgg')

        self.graph = graph
        if pos is None:
            self.pos = nx.spring_layout(graph)
        else:
            self.pos = pos
        self.pivis = pivis
        # Create a figure on screen and set the title
        if dynamic:
            self.graph_ax = plt.axes()
        else:
            self.graph_ax = None
    
    def render(self, *args, **kwargs):
        if self.pivis:
            self.render_pivis_complement(*args, **kwargs)
        else:
            self.render_mpl_coloring(*args, **kwargs)
    
    def render_mpl_coloring(self, coloring: 'Coloring', final=False):
        if self.graph_ax:
            self.graph_ax.clear()
        draw_coloring(graph=self.graph, coloring=coloring, pos=self.pos, legend=True, ax=self.graph_ax)
        if not final:
            plt.pause(0.1)
        else:
            plt.show()
    
    @staticmethod
    def render_pivis_complement(state: 'Coloring', **kwargs):
        _base_size = 10
        s = smooth(x_range=(1, max(state.graph.degree())[1]), y_range=(10, 30))
        nt = Network('500px', '800px', **kwargs)
        # populates the nodes and edges data structures
        graph = state.complement_graph
        g = nx.Graph()
        g.add_edges_from(graph.edges)
        for n in g.nodes:
            # nodes with same color
            if n in graph.color_idxs:
                n_c = set(map(int, graph.nodes[n]['nodes']))
                g.nodes[n]['label'] = f'{n_c}'
                g.nodes[n]['title'] = f'Node group {n_c} color {graph.color_idxs.index(n)}'
                g.nodes[n]['size'] = s(len(n_c))
                g.nodes[n]['group'] = graph.color_idxs.index(n) + 1
            else:
                n_c = n
                g.nodes[n]['label'] = f'{n_c}'
                g.nodes[n]['title'] = f'Lonely node ({n_c})'
                g.nodes[n]['size'] = s(1)
                g.nodes[n]['group'] = 0
            
        nt.from_nx(g)
        return nt.show('complement.html')

    @staticmethod
    def render_pivis_graph(graph, coloring=None, **kwargs):
        nt = Network('500px', '800px', **kwargs)
        # populates the nodes and edges data structures
        
        g = nx.Graph()
        g.add_edges_from(graph.edges)
        
        if coloring:
            for n in g.nodes:
                # nodes with same color
                color = coloring(n)
                color = 0 if color is None else color + 1
                g.nodes[n]['group'] = int(color)
    
        nt.from_nx(g)
        return nt.show('graph.html')