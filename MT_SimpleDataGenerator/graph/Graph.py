from typing import List, Dict
from graph.Node import Node


class Graph:
    def __init__(self):
        self._nodes: List = []

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

    def add_node(self, node_properties: Dict, key: str = None, node_color: str = 'black') -> Node:
        node = Node(node_properties, key, node_color=node_color)
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
