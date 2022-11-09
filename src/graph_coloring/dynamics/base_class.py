import logging

from copy import copy, deepcopy
from typing import Any, List, Generator, Iterator
from itertools import product

import networkx as nx
from scipy.sparse import csr_matrix, diags

from stochopti.discrete_world import space
from gym import Space, Env
from gym.spaces import Discrete
import numpy as np

from graph_coloring.visualizer import Visualizer


class GraphColoringMDPSpace(space.finiteTimeSpace):
    """MDP space representation of the coloring problem."""
    
    def __init__(self, graph: nx.Graph, build_states=True):
        self.graph = graph
        actions = range(len(graph))
        if build_states:
            self.epoch_states = self.get_states()
        else:
            self.epoch_states = [[]]
        super().__init__(sum(self.epoch_states, []), actions, len(graph))
        
        self.T = len(graph)
    
    @staticmethod
    def ds_tour_sequence(state: 'Coloring') -> int:
        return max(state.saturation.keys(), key=lambda x: (state.saturation[x], state.graph.degree[x]))
    
    def build_admisible_actions(self):
        def adm_A(s: Coloring):
            t = sum(len(c) for c in s)
            if t >= len(list(self.graph.nodes)):
                return [None]
            v_t = GraphColoringMDPSpace.ds_tour_sequence(s)
            
            return s.feasible_colors_for_node(v_t)
        
        return adm_A
    
    def build_kernel(self):
        def Q(state: Coloring, action):
            t = len(state.colored_nodes)
            if t >= len(list(self.graph.nodes)):
                return {state: 1}
            v_t = GraphColoringMDPSpace.ds_tour_sequence(state)
            
            s_ = copy(state)
            s_.color_node(v_t, action)
            
            return {s_: 1}
        
        return Q
    
    @staticmethod
    def reward_(state, next_state=None, action=None):
        if action is None:
            return 1
        return len(next_state) - len(state)
    
    def reward(self, state, action=None, time=None):
        if action is None:
            return -1
        s_ = list(self.transition_kernel(state, action).keys())[0]
        return self.reward_(state, s_, action)
    
    def get_states(self):
        initial_sate = Coloring(self.graph)
        initial_sate.color_node(list(self.graph.nodes)[0], 0)
        S = [[initial_sate]]
        
        def succ(s: Coloring):
            succ_ = list()
            t = len(s.colored_nodes)
            v_t = list(self.graph.nodes)[t]
            
            for i, c in enumerate(s):
                s_ = copy(s)
                if s_.color_node(v_t, i, strict=True):
                    succ_.append(s_)
            
            s_ = copy(s)
            s_.color_node(v_t, len(s_))
            succ_.append(s_)
            
            return succ_
        
        for i in range(1, len(self.graph)):
            states = [succ(s) for s in S[i - 1]]
            S.append(sum(states, []))
        
        return S


class GraphColoringActionSpace(Discrete):
    """GYM AI representation of the aqction space for the graph coloring problem."""
    
    def __init__(self, n, env: 'GraphColoringEnv'):
        super().__init__(n)
        self.env = env
    
    def sample(self):
        cur_state = self.env.observation_space.current_state
        feasible_actions = self.env.dynamics.admisible_actions(cur_state)
        
        return self.np_random.choice(feasible_actions)
    
    def feasible_actions(self, state=None):
        if state is None:
            cur_state = self.env.observation_space.current_state
        else:
            cur_state = state
        return self.env.dynamics.admisible_actions(cur_state)


class GraphColoringStateSpace(Space):
    """GYM AI representation of the state space for the graph coloring problem."""
    
    def __init__(self, graph: nx.Graph):
        super().__init__()
        self.graph = graph
        self.current_state = Coloring(graph)
    
    def reset_observation_space(self):
        c = Coloring(self.graph)
        initial_node = list(self.graph.nodes)[0]
        c.color_node(initial_node, 0)
        self.current_state = c
    
    def sample(self):
        epoch = np.random.randint(0, len(self.graph))
        coloring = Coloring(self.graph)
        
        for i in range(epoch):
            colored = False
            node = list(self.graph.nodes)[i]
            while not colored:
                color = np.random.randint(0, len(coloring))
                colored = coloring.color_node(node, color, strict=True)
    
    def contains(self, x):
        if isinstance(x, Coloring):
            return x.conflicting_pairs() == 0
        else:
            return False


class GraphColoringEnv(Env):
    """GYM AI graph coloring class."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.observation_space: GraphColoringStateSpace = GraphColoringStateSpace(graph)
        self.action_space: GraphColoringActionSpace = GraphColoringActionSpace(len(self.graph), self)
        self.dynamics: GraphColoringMDPSpace = GraphColoringMDPSpace(self.graph, build_states=False)
        
        self.visualizer = None
        self._done = False
    
    def simulate_transition_state(self, action, state=None):
        """
        Simulates a state transition without changing the current state.
        
        Parameters
        ----------
        action: int
            The action that wants to be simulated.
        state: Coloring
            The state in which the action wants to be simulated.
        
        Returns
        -------
        new_state
            Returns the state that is achieved by taking action in state.
        
        """
        if state is None:
            state = self.observation_space.current_state
            
        distribution = self.dynamics.transition_kernel(state, action)
        values = list(distribution.keys())
        probabilities = list(distribution.values())
        index = np.random.choice(a=range(len(values)), p=probabilities)
        next_state = values[index]
        reward = self.dynamics.reward_(state, next_state, action)
        return next_state, reward
    
    def step(self, action):
        """
		Given an action takes a step in the specified environment

		Parameters
		----------
		action: int
			An action to be takin in the current state.

		Returns
		-------
		Tuple
			next_state:
			The resulting state of taking the given action in the previous step
			reward:
			The resulting reward of taking the given action in the previous step
			done:
			Boolean flag that indicates if the arrived state is a terminal state
			info:
			Additional information.
		"""
        cur_state = self.observation_space.current_state
        next_state, reward = self.simulate_transition_state(action, state=cur_state)
        
        info = {'state': next_state, 'colored_nodes': next_state.colored_nodes}
        
        if next_state.is_coloring(soft=True):
            done = True
        else:
            done = False
        info['found_coloring'] = done
        
        self.observation_space.current_state = next_state
        self._done = done
        return next_state, reward, done, info
    
    def reset(self):
        """Resets the environment to its initial state."""
        self.observation_space.reset_observation_space()
        self._done = False
        return self.observation_space.current_state
    
    def render(self, mode='human'):
        """Renders the current state."""
        # just raise an exception
        if mode == 'ansi':
            print(self.observation_space.current_state)
        elif mode == 'human':
            if self.visualizer is None:
                self.visualizer = Visualizer(graph=self.graph)
            
            self.visualizer.render(self.observation_space.current_state, final=self._done)
        else:
            super(GraphColoringEnv, self).render(mode=mode)


class Coloring(list):
    """
	Represents the base clas for a coloring of a graph.

	It represents both a partial and a full coloring.


	Attributes
	----------
	graph: nx.Graph
		A pointer to the graph that is coloring
	colored_nodes: set
		A set with the colored nodes.
	"""
    @property
    def saturation(self):
        if self._saturation is None:
            self._saturation = self.complement_graph.compute_saturation()
        return self._saturation
    
    def __init__(self, graph: nx.Graph, complement_graph: nx.Graph = None, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self._coloring = dict()
        self.graph: nx.Graph = graph
        self.colored_nodes = set()
        self.complement_graph = ComplementGraph(copy(self.graph)) if complement_graph is None else complement_graph
        
        self._saturation = None
    
    @classmethod
    def from_matrix_state(cls, state: 'MatrixState'):
        g = nx.Graph(diags(state.matrix.diagonal()).toarray() - state.matrix)
        obj = cls(g)
        for k, v in state.nodes_to_row.items():
            if v in state.color_idxs:
                obj.color_node(k, state.color_idxs.index(v))
        return obj
    
    def color_node(self, node: Any, color: int, strict=False):
        """

		Parameters
		----------
		node: Any
			The node that wants to be colored.
		color: int
			The color that wants to be used for the given node.
		strict: bool
			Boolean flag that when set to True checks if the node can actually be colored with the given color.
			If False, the node will be colored anyway.
		"""
        if strict:
            return self._color_node_strict(node, color)
        else:
            return self._color_node_soft(node, color)
    
    def __call__(self, node):
        return self._coloring.get(node, None)
    
    def __copy__(self):
        cls = self.__class__
        obj = cls(self.graph, complement_graph=deepcopy(self.complement_graph))
        obj._coloring = copy(self._coloring)
        obj.colored_nodes = copy(self.colored_nodes)
        obj._saturation = copy(self._saturation)
        
        for i in self:
            obj.append(copy(i))
        return obj
    
    def __hash__(self):
        return hash(f'{self.graph}-{self}')
    
    def get_color(self, node):
        """Returns the color of the given node."""
        return self(node)
    
    def conflicting_pairs(self):
        """
		Returns the conflicting paris in the coloring, if any.

		Returns
		-------
		List
			A list with the conflicting pairs in the specified coloring, i.e., adjacent nodes that have the same color.
		"""
        c_pairs = []
        for same_color in self:
            for (i, j) in product(same_color, same_color):
                if (i, j) in self.graph.edges:
                    c_pairs.append((i, j))
        return c_pairs
    
    def is_coloring(self, soft=False):
        """
		Checks if the current function is an actual coloring.

		Parameters
		----------
		soft: bool
			Boolean flag that when set to False only checks if it is a partially colored graph.
			Otherwise checks that the number of conflicting pairs equals to zero.

		Returns
		-------
		bool
			True if the specified function is a coloring for graph, False otherwise.
		"""
        if soft:
            return len(self.colored_nodes) == len(self.graph)
        else:
            is_partition = set(self._coloring.keys()) == set(self.graph.nodes)
            zero_conflict = len(self.conflicting_pairs()) == 0
            
            return is_partition and zero_conflict
    
    def number_of_colors(self):
        """Returns the number of used colors"""
        if self.is_coloring():
            return len(self)
        else:
            return float('inf')
    
    def check_if_node_can_be_colored(self, node: Any, color: int):
        """Check if the given node can be colored with color."""
        return self.complement_graph.is_valid_color(node, color)
    
    def feasible_colors_for_node(self, node: Any) -> Iterator[int]:
        """Returns a list with the colors that could be used to color teh given node."""
        yield from self.complement_graph.feasible_colors_for_node(node)
    
    def feasible_node_color_pairs(self) -> Iterator[int]:
        """Returns a list with the colors that could be used to color teh given node."""
        yield from self.complement_graph.feasible_node_color_pairs()
    
    def _color_node_strict(self, node: Any, color: int):
        if self.check_if_node_can_be_colored(node, color):
            return self._color_node_soft(node, color)
        else:
            return False
    
    def _color_node_soft(self, node: Any, color: int):
        cur_color = self._coloring.get(node, None)
        if cur_color is not None:
            self[cur_color].remove(node)
        else:
            self.colored_nodes.add(node)
        
        self._coloring[node] = color
        if len(self) <= color:
            self.append({node})
        else:
            self[color].add(node)        
        self.complement_graph.color_node(node, color, False)
        
        # Todo: update saturation
        self._saturation = None
        return True


class ComplementGraph(nx.Graph):
    
    @property
    def laplacian(self):
        if self._laplacian is None:
            self._laplacian = nx.linalg.laplacian_matrix(self)
        return self._laplacian
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stores the rows in which there are colors
        self.color_idxs = []
        # Stores for each node which row represents it.
        self.nodes_to_row = {n: n for n in self.nodes}
        
        self._laplacian = None
    
    def color_node(self, node_to_color: Any, color_idx: int, strict: bool=True):
        node_i = self.nodes_to_row[node_to_color]
        node_j = node_i if color_idx >= len(self.color_idxs) else self.color_idxs[color_idx]

        if node_i in self.color_idxs:
            logging.warning(f'Node has already been colored...')
            return 0

        if strict:
            if self.is_valid_color(node_to_color, color_idx):
                logging.warning(f'Infeasible to color node {node_to_color} with color {color_idx}.\n'
                                f'There exists adjacent vertexes among nodes that have already been colored with '
                                f'that color.')
                return 0

        if color_idx >= len(self.color_idxs):
            # connect it with all color nodes
            for u in self.color_idxs:
                self.add_edge(u, node_i)
            # color node with new color
            self.color_idxs.append(node_i)
            self.nodes[node_i]['nodes'] = [node_i]
        else:
            # move all arcs from node_j to node_i and delete node_j
            for u in self.neighbors(node_i):
                self.add_edge(node_j, u)
            self.nodes[node_j]['nodes'].append(node_i)            
            self.nodes_to_row[node_i] = node_j
            self.remove_node(node_i)
            
        self._laplacian = None
    
    def is_valid_color(self, node_to_color: Any, color_idx: int):
        if color_idx >= len(self.color_idxs):
            return True
        else:
            node_i = self.nodes_to_row[node_to_color]
            node_j = node_i if color_idx >= len(self.color_idxs) else self.color_idxs[color_idx]
            return not (node_i, node_j) in self.edges

    def feasible_colors_for_node(self, node: Any) -> Iterator[int]:
        idx = self.nodes_to_row[node]
        if idx not in self.color_idxs:
            yield len(self.color_idxs)
            for c in self.color_idxs:
                if (idx, c) not in self.edges:
                    yield self.color_idxs.index(c)

    def feasible_node_color_pairs(self) -> Iterator[int]:
        for node in self.nodes_to_row.keys():
            for_node = self.feasible_colors_for_node(node)
            while True:
                c_ = next(for_node, None)
                if c_ is not None:
                    yield node, c_
                else:
                    break

    def compute_saturation(self):
        def satur(node):
            node_i = self.nodes_to_row[node]
            if node_i in self.color_idxs:
                return -1
            else:
                return len(set(self.neighbors(node)).intersection(self.color_idxs))
        return {n: satur(n) for n in self.nodes}
