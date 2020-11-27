from typing import List, Dict
from graph.Node import Node
import random
import pandas as pd
import pickle
import math
import stellargraph as sg


class Graph:
    def __init__(self, name: str):
        self._name: str = name
        self._nodes: List = []

    def get_name(self) -> str:
        return self._name

    def get_node_by_index(self, index: int) -> Node:
        return self._nodes[index]

    def get_node_by_id(self, id: int) -> Node:
        for node in self._nodes:
            if node.get_id() == id:
                return node

    def get_node_by_key(self, key: str) -> Node:
        for node in self._nodes:
            if node.get_key() == key:
                return node

    def get_random_node(self) -> Node:
        return self._nodes[random.randint(0, len(self._nodes) - 1)]

    def add_node(self, node_properties: Dict, key: str = None, node_color: str = 'black', node_type: str = '') -> Node:
        node = Node(node_properties, key, node_color=node_color, node_type=node_type)
        self._nodes.append(node)
        return node

    def remove_node(self, node: Node):
        node.remove_from_neighbors()
        self._nodes.remove(node)

    def prune(self, min_cluster_size: int = 3):
        for node in reversed(self._nodes):
            if self.compute_cluster_size(node) < min_cluster_size:
                self.remove_node(node)

        return self

    def reset_found_nodes(self) -> None:
        for node in self._nodes:
            node.set_is_found(False)

    def compute_cluster_size(self, starting_node: Node) -> int:
        self.reset_found_nodes()

        return self.compute_cluster_size_recursive(starting_node)

    def compute_cluster_size_recursive(self, root_node: Node) -> int:
        if root_node.get_is_found():
            return 0

        root_node.set_is_found(True)
        cluster_size_neighbors = 0

        for neighbor in (root_node.get_neighbors() + root_node.get_references()):
            cluster_size_neighbors += self.compute_cluster_size_recursive(neighbor)

        return 1 + cluster_size_neighbors

    def export_graphviz(self, filename: str, drop_isolated_nodes: bool = True) -> None:
        node_string = ''
        for node in self._nodes:
            if not drop_isolated_nodes or len(node.get_neighbors()) > 0 or len(node.get_references()) > 0: # todo make pruning a method of the graph and run before classification
                # todo add to each node: get_subgraph_size(node) -> count dfs connected nodes
                # todo for this use node.marked property
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

    # https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph

    def serialize_stellargraph(self, attributes: List[str], node_types: List[str]) -> (sg.StellarDiGraph, bool, str):
        # nodes_gt = pd.Series()
        contains_fraud = False
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
            if 'is_fraud' in node_properties and node_properties['is_fraud']:
            # if node.get_type() == 'MARKED':
            #     for neighbor in node.get_neighbors():
            #         if neighbor.get_type() == 'MARKED':
                        contains_fraud = True

            # data
            for attribute_name in attributes:
                if attribute_name in node_properties and (isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name], float)):
                    nodes[attribute_name].append(math.log(node_properties[attribute_name]))
                else:
                    nodes[attribute_name].append(0)

            for type_name in node_types:
                nodes[f'type_{type_name}'].append(node.get_type() == type_name)

            for neighbor in node.get_neighbors():
                edges['source'].append(node.get_id())
                edges['target'].append(neighbor.get_id())

        return sg.StellarDiGraph( pd.DataFrame(nodes, index=nodes_index), edges=pd.DataFrame(edges)), contains_fraud, self._name

    def serialize_stellargraph_node_level(self, attributes: List[str], node_types: List[str]) -> (sg.StellarDiGraph, pd.Series):
        nodes_gt = pd.Series()
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
                nodes_gt._set_value(node.get_id(), 'fraud' if node_properties['is_fraud'] else 'no_fraud')
            else:
                nodes_gt._set_value(node.get_id(), 'no_fraud')

            # data
            for attribute_name in attributes:
                if attribute_name in node_properties and (
                        isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name],
                                                                                       float)):
                    nodes[attribute_name].append(math.log(node_properties[attribute_name]))
                else:
                    nodes[attribute_name].append(0)

            for type_name in node_types:
                nodes[f'type_{type_name}'].append(node.get_type() == type_name)

            for neighbor in node.get_neighbors():
                edges['source'].append(node.get_id())
                edges['target'].append(neighbor.get_id())

        return sg.StellarDiGraph(pd.DataFrame(nodes, index=nodes_index), edges=pd.DataFrame(edges)), nodes_gt, self._name

    # def serialize_stellargraph(self, attributes: List[str]) -> (sg.StellarDiGraph, pd.Series):
    def serialize_stellargraph_with_node_types(self, attributes: List[str]) -> (sg.StellarDiGraph, bool):
        # nodes_gt = pd.Series()
        contains_fraud = False
        edges = {
            'source': [],
            'target': []
        }
        nodes = {}
        nodes_index = {}

        # for attribute_name in attributes:
        #     nodes[attribute_name] = []

        for index, node in enumerate(self._nodes):
            node_properties = node.get_properties()

            # ground truth
            if 'is_fraud' in node_properties and node_properties['is_fraud']:
                contains_fraud = True
            # if 'is_fraud' in node_properties:
            #     nodes_gt._set_value(node.get_id(), 'fraud' if node['is_fraud'] else 'no_fraud')
            # else:
            #     nodes_gt._set_value(node.get_id(), 'no_fraud')

            # data
            if node.get_type() not in nodes:
                nodes[node.get_type()] = {}
                nodes_index[node.get_type()] = []

            nodes_index[node.get_type()].append(node.get_id())

            for attribute_name in attributes:
                if attribute_name not in nodes[node.get_type()]:
                    nodes[node.get_type()][attribute_name] = []

                if attribute_name in node_properties and (isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name], float)):
                    nodes[node.get_type()][attribute_name].append(node_properties[attribute_name])
                else:
                    nodes[node.get_type()][attribute_name].append(0)

            for neighbor in node.get_neighbors():
                edges['source'].append(node.get_id())
                edges['target'].append(neighbor.get_id())

        sg_nodes = {}
        for key in nodes:
            sg_nodes[key] = pd.DataFrame(nodes[key], index=nodes_index[key])

        # return sg.StellarDiGraph(sg_nodes, edges=pd.DataFrame(edges)), nodes_gt
        return sg.StellarDiGraph(sg_nodes, edges=pd.DataFrame(edges)), contains_fraud

    def __len__(self):
        return len(self._nodes)
