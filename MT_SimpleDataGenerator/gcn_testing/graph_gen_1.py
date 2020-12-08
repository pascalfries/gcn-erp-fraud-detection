import random
from graph.Graph import Graph
from graph.GraphCollection import GraphCollection
from helpers import with_probability, rand_float
import config as cfg


def generate_graph(num) -> Graph:
    graph = Graph(f'Graph_{num}')
    # product_count = random.randint(2, 10)
    product_count = 1000

    for n in range(product_count):
        price = random.randint(5, 500)
        product_node = graph.add_node({'type': 'PRODUCT', 'price': price}, f'Product_{n}', node_type='PRODUCT')

        if with_probability(0.04):
            new_price_rate = rand_float(0.65, 0.8)
            change_node = graph.add_node({'type': 'CHANGE', 'old_price': price, 'new_price': price * new_price_rate, 'is_fraud': True}, f'Change_of_Prod_{n}', node_type='CHANGE', node_color='red')
            change_node.add_neighbor(product_node)
        elif with_probability(0.45):
            new_price_rate = rand_float(0.8, 1)
            change_node = graph.add_node({'type': 'CHANGE', 'old_price': price, 'new_price': price * new_price_rate, 'is_fraud': False}, f'Change_of_Prod_{n}', node_type='CHANGE', node_color='green')
            change_node.add_neighbor(product_node)

    # for node in graph._nodes:
    #     link_count = random.randint(0, min(8, int(node_count / 4)))
    #
    #     for _ in range(link_count):
    #         node.add_neighbor(graph.get_random_node())

    return graph


# MAIN CODE
graphs = GraphCollection()
graphs.add_graph(generate_graph(0))

graphs.save(cfg.STORAGE_ROOT_PATH + r'\gcn_testing')

with open(cfg.STORAGE_ROOT_PATH + '\\gcn_testing_viz\\generate_graphs.bat', 'w') as graphviz_script:
    for index, graph in enumerate(graphs.get_raw_list()):
        graph.export_graphviz(cfg.STORAGE_ROOT_PATH + rf'\\gcn_testing_viz\{graph.get_name()}.txt', drop_isolated_nodes=False)
        print(f'dot -Tsvg {graph.get_name()}.txt -o out/graph_{graph.get_name()}.svg', file=graphviz_script)
