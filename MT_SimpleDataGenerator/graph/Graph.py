from typing import List, Dict
from graph.Node import Node
import random
import pandas as pd
import numpy as np
import pickle
import math
import stellargraph as sg


class Graph:
    def __init__(self, name: str):
        self._name: str = name
        self._nodes: List = []
        self._nodes_by_key: Dict = {}

    def get_name(self) -> str:
        return self._name

    def get_node_by_id(self, id: int) -> Node:
        for node in self._nodes:
            if node.get_id() == id:
                return node

    def get_node_by_index(self, index: int) -> Node:
        return self._nodes[index]

    def get_node_by_key(self, key: str) -> Node:
        return self._nodes_by_key.get(key, None)

    def get_random_node(self) -> Node:
        return self._nodes[random.randint(0, len(self._nodes) - 1)]

    def add_node(self, node_properties: Dict, key: str = None, node_color: str = 'black', node_type: str = '') -> Node:
        node = Node(node_properties, key, node_color=node_color, node_type=node_type)
        self._nodes.append(node)
        self._nodes_by_key[key] = node
        return node

    def remove_node(self, node: Node):
        node.remove_from_neighbors()
        del self._nodes_by_key[node.get_key()]
        self._nodes.remove(node)

    def prune(self, min_cluster_size: int = 3):
        for node in reversed(self._nodes):
            if node.get_is_part_of_large_cluster():
                continue

            cluster_size, cluster_nodes = self.compute_cluster(node)

            if cluster_size < min_cluster_size:
                self.remove_node(node)
            else:
                for cluster_node in cluster_nodes:
                    cluster_node.set_is_part_of_large_cluster(True)

        return self

    def reset_found_nodes(self) -> None:
        for node in self._nodes:
            node.set_is_found(False)

    def compute_cluster(self, starting_node: Node) -> (int, List[Node]):
        self.reset_found_nodes()

        return self.compute_cluster_recursive(starting_node)

    def compute_cluster_recursive(self, root_node: Node) -> (int, List[Node]):
        if root_node.get_is_found():
            return 0, []

        root_node.set_is_found(True)
        cluster_size_neighbors = 0
        cluster_nodes = [root_node]

        for neighbor in (root_node.get_neighbors() + root_node.get_references()):
            size, nodes = self.compute_cluster_recursive(neighbor)
            cluster_size_neighbors += size
            cluster_nodes += nodes

        return (1 + cluster_size_neighbors), cluster_nodes

    def export_graphviz(self, filename: str, drop_isolated_nodes: bool = True) -> None:
        node_string = ''
        for node in self._nodes:
            if not drop_isolated_nodes or len(node.get_neighbors()) > 0 or len(node.get_references()) > 0:
                node_string += node.generate_graphviz()

        f = open(filename, 'w')
        f.write('digraph G {\nnode [shape=box,color=black,fontname=Arial,labelloc=c];\nedge [color=gray50,style=bold];\n\n' + node_string + '\n}')
        f.close()

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as graph_file:
            pickle.dump(self, graph_file)

    def serialize_numpy(self, attributes: List[str]) -> (List, List):
        graph_nodes = []
        graph_edges = []

        for index, node in enumerate(self._nodes):
            node_properties = node.get_properties()

            row = [node.get_id()]

            for attribute_name in attributes:
                if attribute_name in node_properties and (isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name], float)):
                    row.append(node_properties[attribute_name])
                else:
                    row.append(0)

            row.append('fraud' if 'is_fraud' in node_properties and node_properties['is_fraud'] else 'no_fraud')
            graph_nodes.append(row)

            for neighbor in node.get_neighbors():
                graph_edges.append([neighbor.get_id(), node.get_id()])
                # graph_edges.append([node.get_id(), neighbor.get_id()])

        return graph_nodes, graph_edges

    def serialize_stellargraph(self, attributes: List[str], node_types: List[str]) -> (sg.StellarDiGraph, bool, str, set):
        contains_fraud = False
        edges = {
            'source': [],
            'target': []
        }
        nodes = {}
        nodes_index = []
        fraud_ids = set()

        for attribute_name in attributes:
            nodes[attribute_name] = []

        for type_name in node_types:
            nodes[f'type_{type_name}'] = []

        for index, node in enumerate(self._nodes):
            node_properties = node.get_properties()
            nodes_index.append(node.get_id())

            # ground truth
            if 'is_fraud' in node_properties and node_properties['is_fraud']:
                contains_fraud = True

                fraud_ids.add(node_properties['fraud_id'])

            # data
            for attribute_name in attributes:
                if attribute_name in node_properties:
                    if isinstance(node_properties[attribute_name], bool):
                        nodes[attribute_name].append(1 if node_properties[attribute_name] else 0)
                    elif isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name], float):
                        nodes[attribute_name].append(node_properties[attribute_name])
                    elif isinstance(node_properties[attribute_name], str) and node_properties[attribute_name].isdigit():
                        nodes[attribute_name].append(float(node_properties[attribute_name]))
                    else:
                        nodes[attribute_name].append(np.nan)
                else:
                    nodes[attribute_name].append(np.nan)

            for type_name in node_types:
                nodes[f'type_{type_name}'].append(node.get_type() == type_name)

            for neighbor in node.get_neighbors():
                edges['source'].append(node.get_id())
                edges['target'].append(neighbor.get_id())

        nodes_df = pd.DataFrame(nodes, index=nodes_index)

        # normalize
        for attribute_name in attributes:
            column_data = nodes_df[attribute_name]
            nodes_df[attribute_name] = ((column_data - np.nanmean(column_data)) / np.nanstd(column_data))

        nodes_df.fillna(-1, inplace=True)

        return sg.StellarDiGraph(nodes_df, edges=pd.DataFrame(edges)), contains_fraud, self._name, fraud_ids

    def serialize_stellargraph_node_level(self, attributes: List[str], node_types: List[str]) -> (sg.StellarDiGraph, pd.Series, pd.Series):
        nodes_gt = pd.Series()
        nodes_fraud_id = pd.Series()
        edges = {
            'source': [],
            'target': []
        }
        nodes = {}
        nodes_index = []

        for attribute_name in attributes:
            nodes[attribute_name] = []

        for type_name in node_types:
            nodes[f'type_{type_name}'] = []

        for index, node in enumerate(self._nodes):
            node_properties = node.get_properties()
            nodes_index.append(node.get_id())

            # ground truth
            if 'is_fraud' in node_properties:
                if node_properties['is_fraud']:
                    nodes_gt._set_value(node.get_id(), 'fraud')
                    nodes_fraud_id._set_value(node.get_id(), node_properties['fraud_id'])
                else:
                    nodes_gt._set_value(node.get_id(), 'no_fraud')
                    nodes_fraud_id._set_value(node.get_id(), None)
            else:
                nodes_gt._set_value(node.get_id(), 'irrelevant')
                nodes_fraud_id._set_value(node.get_id(), None)

            # attributes
            for attribute_name in attributes:
                if attribute_name in node_properties:
                    if isinstance(node_properties[attribute_name], bool):
                        nodes[attribute_name].append(1 if node_properties[attribute_name] else 0)
                    elif isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name], float):
                        nodes[attribute_name].append(node_properties[attribute_name])
                    elif isinstance(node_properties[attribute_name], str) and node_properties[attribute_name].isdigit():
                        nodes[attribute_name].append(float(node_properties[attribute_name]))
                    else:
                        nodes[attribute_name].append(np.nan)
                else:
                    nodes[attribute_name].append(np.nan)

            for type_name in node_types:
                nodes[f'type_{type_name}'].append(node.get_type() == type_name)

            for neighbor in node.get_neighbors():
                edges['source'].append(node.get_id())
                edges['target'].append(neighbor.get_id())

        nodes_df = pd.DataFrame(nodes, index=nodes_index)

        # normalize
        for attribute_name in attributes:
            column_data = nodes_df[attribute_name]
            nodes_df[attribute_name] = ((column_data - np.nanmean(column_data)) / np.nanstd(column_data))

        nodes_df.fillna(-1, inplace=True) # todo check indexes of df

        return sg.StellarDiGraph(nodes_df, edges=pd.DataFrame(edges)), nodes_gt, self._name, nodes_fraud_id

    def __len__(self):
        return len(self._nodes)
