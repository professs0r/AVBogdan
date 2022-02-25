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

directed_adjacency_matrix = np.array([(0,1,0,0,0),
                                      (0,0,1,0,1),
                                      (0,0,0,1,1),
                                      (1,0,0,0,0),
                                      (1,0,0,1,0)])

conductivity_matrix = np.zeros((COUNT_NODES-1, COUNT_NODES-1))
current_matrix = np.zeros((COUNT_NODES-1, 1))
#potential_matrix = current_matrix / conductivity_matrix

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

# temporary initialization
zero_potential = 4

export_array = []
import_array = []
for potential in range(COUNT_NODES): # за данный проход формируется уравнение узловых потенциалов относительно рассматриваемого узла
    print("potential ", potential + 1, ":\n")
    if (potential == zero_potential):
        continue
    for node in range(COUNT_NODES):
        if (potential == node):
            for export_node in range(len(directed_adjacency_matrix[node])):
                if(directed_adjacency_matrix[node][export_node] != 0):
                    export_array.append(export_node)
        else:
            if(directed_adjacency_matrix[node][potential] == 1):
                import_array.append(node)
    for index_matrix in range(len(conductivity_matrix[potential])):
        print(0)
        if (index_matrix == zero_potential):
            continue
        if (potential == index_matrix):
            if (len(export_array) == 1):
                conductivity_matrix[potential][index_matrix] += 1/(graph[potential][export_array[0]]['resistance'])
            else:
                for exp_arr in range(len(export_array)):
                    conductivity_matrix[potential][index_matrix] += 1 / (graph[potential][export_array[exp_arr]]['resistance'])
            if (len(import_array) == 1):
                conductivity_matrix[potential][index_matrix] += 1 / (graph[import_array[0]][potential]['resistance'])
            else:
                for imp_arr in range(len(import_array)):
                    conductivity_matrix[potential][index_matrix] += 1 / (graph[import_array[imp_arr]][potential]['resistance'])
        else:
            # if zepo_potential нужно

        #conductivity_matrix[potential][index_matrix] = # temp note
    print("Конец итерации номер: ", potential + 1)
    export_array.clear()
    import_array.clear()




index = 1
#for branch in graph.edges():
    #print("itteration: ", index)
    #index += 1
    #graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] - graph.nodes[branch[1]]['potential'] + graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]]['resistance']
    #print("branch[", branch[0], "]][branch[", branch[1], "]]['I'] = (", graph.nodes[branch[0]], " - ", graph.nodes[branch[1]], " + ", graph[branch[0]][branch[1]]['voltage'], ") / ", graph[branch[0]][branch[1]]['resistance'])
    #print(graph.nodes[4])
    #print(graph[branch[0]][branch[1]]['resistance'])
