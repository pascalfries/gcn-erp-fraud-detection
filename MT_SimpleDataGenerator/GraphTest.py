from graph.Graph import Graph
from graph.Node import Node


g = Graph()

n1 = g.add_node({'table': 'TABLE 1', 'foo': 'bar', 'xx': 'yy'})
n2 = g.add_node({'table': 'TABLE 2'})
n3 = g.add_node({'table': 'TABLE 3', 'foo': 'bar'})
n4 = g.add_node({'table': 'TABLE 4'})
n5 = g.add_node({'table': 'TABLE 5'})

n1.add_neighbor(n2)
n1.add_neighbor(n3)
n4.add_neighbor(n3)
n5.add_neighbor(n4)
n4.add_neighbor(n1)

g.export_graphviz(r'C:\Users\Pasi\Desktop\mt\graph.txt')
