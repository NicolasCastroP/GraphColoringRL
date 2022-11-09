from graph_coloring.dynamics.base_class import GraphColoringEnv
import networkx as nx


def random_run(graph):
    env = GraphColoringEnv(graph)
    done = False
    state = env.reset()
    
    while not done:
        env.render(mode='human')
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
    
    env.render(mode='human')
    return state


if __name__ == '__main__':
    g = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)
    random_run(g)
