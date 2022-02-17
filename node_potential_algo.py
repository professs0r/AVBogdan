import numpy as np
import networkx as nx
import math

COUNT_NODES = 5
COUNT_BRANCHES = 8

X_COORDINATES = [1, 25, 45, 65, 35]
Y_COORDINATES = [1, 5, 5, 5, 15]

directed_adjacency_list = np.array([(1),
                                    (2,4),
                                    (3,4),
                                    (0),
                                    (0,3)])

def func_initialization(list, nodes, branches):
    graph = nx.DiGraph()
    for index in range(nodes):
        graph.add_node(index, potential=index)
    graph.add_edge(0, 1, resistance=0.1, voltage=630, type='СИП', length=1000, I=0)
    graph.add_edge(1, 2, resistance=1.17, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(1, 4, resistance=2.87, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 3, resistance=2.55, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 4, resistance=0.84, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(3, 0, resistance=70.1, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(4, 0, resistance=70, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(4, 3, resistance=3.69, voltage=0, type='СИП', length=1000, I=0)
    return graph

graph = func_initialization(directed_adjacency_list, COUNT_NODES, COUNT_BRANCHES)
index = 1
for branch in graph.edges():
    print("itteration: ", index)
    index += 1
    graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] - graph.nodes[branch[1]]['potential'] + graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]]['resistance']
    print("branch[", branch[0], "]][branch[", branch[1], "]]['I'] = (", graph.nodes[branch[0]], " - ", graph.nodes[branch[1]], " + ", graph[branch[0]][branch[1]]['voltage'], ") / ", graph[branch[0]][branch[1]]['resistance'])
    #print(graph.nodes[4])
    #print(graph[branch[0]][branch[1]]['resistance'])
