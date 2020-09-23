from typing import List, Optional
from graph.Graph import Graph
import pickle
import os


class GraphCollection:
    def __init__(self, graphs: Optional[List[Graph]] = None):
        if graphs is None:
            graphs = []

        self._graphs = graphs

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
