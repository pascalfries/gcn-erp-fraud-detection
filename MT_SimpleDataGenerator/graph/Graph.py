from typing import List, Dict
from graph.Node import Node
import numpy as np
import pandas as pd
import pickle
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

    def add_node(self, node_properties: Dict, key: str = None, node_color: str = 'black', node_type: str = '') -> Node:
        node = Node(node_properties, key, node_color=node_color, node_type=node_type)
        self._nodes.append(node)
        return node

    def export_graphviz(self, filename: str) -> None:
        node_string = ''
        for node in self._nodes:
            if len(node.get_neighbors()) > 0 or len(node.get_references()) > 0:
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

    def serialize_stellargraph(self, attributes: List[str]) -> (sg.StellarDiGraph, bool):
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

        for index, node in enumerate(self._nodes):
            node_properties = node.get_properties()
            nodes_index.append(node.get_id())

            # ground truth
            if 'is_fraud' in node_properties and node_properties['is_fraud']:
                contains_fraud = True

            # data
            for attribute_name in attributes:
                if attribute_name in node_properties and (isinstance(node_properties[attribute_name], int) or isinstance(node_properties[attribute_name], float)):
                    nodes[attribute_name].append(node_properties[attribute_name])
                else:
                    nodes[attribute_name].append(0)

            for neighbor in node.get_neighbors():
                edges['source'].append(node.get_id())
                edges['target'].append(neighbor.get_id())

        return sg.StellarDiGraph( pd.DataFrame(nodes, index=nodes_index), edges=pd.DataFrame(edges)), contains_fraud

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
