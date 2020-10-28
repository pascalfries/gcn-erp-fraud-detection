from typing import List, Optional, Tuple
from graph.Graph import Graph
import numpy as np
import pandas as pd
import pickle
import os
import stellargraph as sg


class GraphCollection:
    def __init__(self, graphs: Optional[List[Graph]] = None, directory_name: Optional[str] = None):
        if graphs is None:
            graphs = []

        self._graphs = graphs

        if directory_name is not None:
            self.load(directory_name)

    def add_graph(self, graph: Graph):
        self._graphs.append(graph)

    def get_raw_list(self) -> List[Graph]:
        return self._graphs

    def save(self, directory_path: str):
        for graph in self._graphs:
            graph.save(directory_path + '/' + graph.get_name() + '.graph')

    def load(self, directory_name: str):
        for file_name in os.listdir(directory_name):
            if file_name.endswith('.graph'):
                with open(directory_name + '/' + file_name, 'rb') as graph_file:
                    self._graphs.append(pickle.load(graph_file))

    def save_numpy(self, directory_path: str, attributes: List[str]):
        for graph in self._graphs:
            nodes, edges = graph.serialize_numpy(attributes)

            np.savetxt(directory_path + graph.get_name() + '.vertices', nodes, fmt='%s')
            np.savetxt(directory_path + graph.get_name() + '.edges', edges)

    def serialize_stellargraph(self, attributes: List[str]) -> List[Tuple[sg.StellarDiGraph, pd.Series]]:
        return list(map(lambda graph: graph.serialize_stellargraph(attributes), self._graphs))

    # def serialize_numpy(self, attributes: List[str]) -> (np.array, np.array):
    #     nodes = []
    #     edges = []
    #
    #     for graph in self._graphs:
    #         graph_nodes, graph_edges = graph.serialize_numpy(attributes)
    #
    #         nodes.extend(graph_nodes)
    #         edges.extend(graph_edges)
    #
    #     return np.array(nodes), np.array(edges)
