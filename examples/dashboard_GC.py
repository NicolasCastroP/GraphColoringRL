from abc import abstractmethod
from ipywidgets import widgets
from matplotlib import pyplot as plt

from graph_coloring import Visualizer, SpectralBound, GreedyColoring, RandomRolloutLB, RolloutLB
from graph_coloring.graph_generator import *


class _SectionBase:
    OUT_LAYOUT = widgets.Layout(
        height='1000px', width='1000px', margin='0 50px 0 50px', padding='0 20px 0 20px',
        # justify_content='center',
        # justify_items='center',
        # align_content='center'
    )

    def __init__(self):
        self.widgets = {}
        self.out = widgets.Output()

        self.build_widgets()
        self.link_widgets()

    @abstractmethod
    def build_widgets(self):
        ...

    def link_widgets(self):
        for _, w in self.widgets.items():
            w.observe(self.on_change)

    def display(self):
        out = widgets.VBox([
            widgets.HBox(list(self.widgets.values())),
            self.out
        ], layout=_SectionBase.OUT_LAYOUT)

        self.update_output()
        return out

    @abstractmethod
    def update_output(self):
        ...

    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.update_output()
            

class GraphGeneration(_SectionBase):
    def __init__(self):
        super().__init__()
        self.graph = None
        self.coloring = None
        
    def build_widgets(self):
        self.widgets['num_nodes'] = widgets.IntSlider(
            value=10, min=10, max=100, description='# Nodes'
        )

        self.widgets['num_colors'] = widgets.IntSlider(
            value=7, min=4, max=20, description='# Colors'
        )
        
        self.widgets['colored'] = widgets.ToggleButton(
            value=False,
            description='Colored',
            icon='aperture'
        )

        self.widgets['plot'] = widgets.ToggleButton(
            value=True,
            description='Plot',
            icon='aperture'
        )

    def update_output(self):
        num_nodes = self.widgets['num_nodes'].value
        num_colors = self.widgets['num_colors'].value
    
        self.generate_graph(num_nodes, num_colors)
        
        colored = self.widgets['colored'].value
        with self.out:
            self.out.clear_output(True)
            if self.widgets['plot'].value:
                self.plot_graph(colored)
    
    def generate_graph(self, num_nodes, num_colors, **kwargs):
        self.graph = generate_graph(number_of_nodes=num_nodes, number_of_colors=num_colors, **kwargs)
        self.coloring = create_coloring(self.graph)
        
    def plot_graph(self, colored):
        coloring = self.coloring if colored else None
        display(Visualizer.render_pivis_graph(self.graph, coloring, notebook=True))
    

class ComplementGraph(_SectionBase):
    def __init__(self, graph_generator: GraphGeneration):
        self.graph_generator = graph_generator
        g = nx.Graph()
        g.add_edges_from(self.graph_generator.graph.edges)
        self.graph = g
        self.coloring = Coloring(self.graph)
        self.nodes = sorted(list(self.graph.nodes))
        self.colors = range(len(self.coloring) + 1)
        
        self.lb = SpectralBound(self.graph)
        self.lb_list = [self.lb(self.coloring)]
        super().__init__()
        
        self.out2 = widgets.Output()
        self.out3 = widgets.Output()
        
    def build_widgets(self):
        self.widgets['node'] = widgets.Dropdown(
            options=self.nodes,
            value=self.nodes[0],
            description='Select Node:',
            disabled=False,
        )
        
        self.widgets['color'] = widgets.Dropdown(
            options=self.colors,
            value=self.colors[0],
            description='Color:',
            disabled=False,
        )
        
        self.widgets['color_node'] = widgets.Button(
            description='Color Node'
        )

        self.widgets['reset'] = widgets.Button(
            description='Reset'
        )

        self.widgets['lb'] = widgets.ToggleButton(
            value=False,
            description='Plot LB',
            icon='aperture'
        )
    
    def link_widgets(self):
        self.widgets['color_node'].on_click(self.on_color_node)
        self.widgets['reset'].on_click(self.reset)
    
    def on_color_node(self, b):
        node = self.widgets['node'].value
        color = self.widgets['color'].value
        colored = self.coloring.color_node(node, color, strict=True)
        
        if colored:
            nodes = list(self.widgets['node'].options)
            nodes.remove(node)
            self.widgets['node'].options = nodes
            
            self.colors = range(len(self.coloring) + 1)
    
            self.widgets['color'].options = self.colors
            
            self.lb_list.append(len(self.coloring) + self.lb(self.coloring))
            
            self.update_output()
        
    def update_output(self):
        with self.out:
            self.out.clear_output(True)
            display(Visualizer.render_pivis_graph(self.graph, self.coloring, notebook=True))
        
        with self.out2:
            self.out2.clear_output(True)
            display(Visualizer.render_pivis_complement(self.coloring, notebook=True))
            
        with self.out3:
            self.out3.clear_output(True)
            if self.widgets['lb'].value:
                self.plot_lb()
    
    def display(self):
        out = widgets.VBox([
            widgets.HBox(list(self.widgets.values())),
            widgets.HBox([self.out, self.out2]),
            self.out3
        ], layout=_SectionBase.OUT_LAYOUT)
    
        self.update_output()
        return out

    def reset(self, b=None):
        g = nx.Graph()
        g.add_edges_from(self.graph_generator.graph.edges)
        self.graph = g
        self.coloring = Coloring(self.graph)
        self.nodes = sorted(list(self.graph.nodes))
        self.colors = range(len(self.coloring) + 1)

        self.widgets['node'].options = self.nodes
        self.widgets['color'].options = self.colors

        self.lb = SpectralBound(self.graph)
        self.lb_list = []
        
        self.update_output()

    def plot_lb(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(self.lb_list))
        if len(x):
            opt = len(self.graph_generator.coloring)
            ax.plot(x, self.lb_list, '--', label='Lower Bound')
            ax.hlines(opt, min(x), max(x), linestyles='--', colors='green')
        plt.show()


class RolloutGUI(_SectionBase):
    def __init__(self, complement_graph: ComplementGraph):
        super().__init__()
        self.complement_graph = complement_graph
        g = nx.Graph()
        g.add_edges_from(self.complement_graph.graph.edges)
        self.graph = g

        self.heuristic = GreedyColoring(self.graph)
        self.lower_bound = SpectralBound(self.graph)
        
        self.rollout = self.choose_algorithm()
        self.solved = False

    def build_widgets(self):
        self.widgets['algorithm'] = widgets.Dropdown(
            options=['Rollout LB', 'Random Rollout LB'],
            values='Random Rollout LB'
        )

        self.widgets['approx'] = widgets.Dropdown(
            options=['Heuristic', 'Lower Bound'],
            values='Random Rollout LB'
        )
        
        self.widgets['depth'] = widgets.IntSlider(
            value=2, min=1, max=4, description='Depth'
        )

        self.widgets['fortified'] = widgets.ToggleButton(
            value=False,
            description='Fortified',
            icon='aperture'
        )
        
        self.widgets['reset'] = widgets.Button(
            description='Reset'
        )
        
        self.widgets['run'] = widgets.Button(
            description='Run'
        )

    def update_output(self):
        with self.out:
            self.out.clear_output(True)
            opt = len(self.complement_graph.graph_generator.coloring)
            self.rollout.bounds_plot(optimal_value=opt)
    
    def link_widgets(self):
        self.widgets['reset'].on_click(self.on_reset)
        self.widgets['run'].on_click(self.on_run)
    
    def choose_algorithm(self):
        depth = self.widgets['depth'].value
        algo = self.widgets['algorithm'].value
        approx = self.widgets['approx'].value
        function_approximation = self.heuristic if approx == 'Heuristic' else self.lower_bound
        if algo == 'Rollout LB':
            return RolloutLB(
                graph=self.graph,
                heuristic=self.heuristic,
                lower_bound=self.lower_bound,
                function_approximation=function_approximation,
                depth=depth,
                fortified=self.widgets['fortified'].value
            )
        else:
            return RandomRolloutLB(
                graph=self.graph,
                heuristic=self.heuristic,
                lower_bound=self.lower_bound,
                function_approximation=function_approximation,
                depth=depth,
                fortified=self.widgets['fortified'].value
            )
       
    def on_reset(self, b=None):
        self.complement_graph.reset()

        g = nx.Graph()
        g.add_edges_from(self.complement_graph.graph.edges)
        self.graph = g

        self.heuristic = GreedyColoring(self.graph)
        self.lower_bound = SpectralBound(self.graph)

        self.rollout = self.choose_algorithm()
        self.solved = False
        with self.out:
            self.out.clear_output(True)
    
    def on_run(self, b):
        self.on_reset()
        coloring = self.rollout.solve()
        self.complement_graph.coloring = coloring
        self.complement_graph.update_output()
        self.update_output()
        self.solved = True
        

def main():
    gg = GraphGeneration()
    gg.plot_graph(True)


if __name__ == '__main__':
    main()