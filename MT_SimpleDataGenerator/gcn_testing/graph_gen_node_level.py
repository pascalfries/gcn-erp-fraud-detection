import random
from graph.Graph import Graph
from typing import Tuple, List
import config as cfg
import pandas as pd


def generate_graph() -> Tuple[Graph, pd.Series]:
    graph = Graph(f'Graph Node Level')
    node_count = 25_000
    nodes_gt = pd.Series()

    for n in range(node_count):
        type = random.choice(['type-1', 'type-2', 'type-3'])
        node = graph.add_node({}, f'Node_{n}', node_type=type)

        nodes_gt._set_value(node.get_id(), type)

    for node in graph._nodes:
        link_count = random.randint(0, 2) #random.randint(0, min(7, int(node_count / 4)))

        for _ in range(link_count):
            node.add_neighbor(graph.get_random_node())

    return graph, nodes_gt


# MAIN CODE
graph, gt = generate_graph()

graph.export_graphviz(cfg.STORAGE_ROOT_PATH + rf'\\gcn_testing_viz\{graph.get_name()}.txt', drop_isolated_nodes=False)
