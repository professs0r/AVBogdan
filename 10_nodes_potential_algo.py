import numpy as np
import networkx as nx

# start source data

COUNT_NODES = 10
COUNT_BRANCHES = 17

directed_adjacency_list = np.array([1,
                                    (2, 4),
                                    (3, 5),
                                    (0, 6),
                                    (5, 7),
                                    (0, 6, 8),
                                    9,
                                    (0, 8),
                                    9,
                                    0])

def func_list_to_matrix(adjacency_list):
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)))
    for row in range(len(adjacency_list)):
        if (isinstance(adjacency_list[row], int)):
            adjacency_matrix[row][adjacency_list[row]] = 1
        else:
            for column in range(len(adjacency_list[row])):
                adjacency_matrix[row][adjacency_list[row][column]] = 1
    return adjacency_matrix

directed_adjacency_matrix = func_list_to_matrix(directed_adjacency_list)

def func_initialization(list, nodes, branches):
    graph = nx.DiGraph()
    for index in range(nodes):
        graph.add_node(index, potential=0)
    graph.add_edge(0, 1, resistance=0.1, voltage=630, type='СИП', length=1000, I=0)
    graph.add_edge(1, 2, resistance=1.17, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(1, 4, resistance=1.35, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 3, resistance=2.55, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 5, resistance=2.01, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(3, 0, resistance=70.1, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(3, 6, resistance=1.68, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(4, 5, resistance=1.25, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(4, 7, resistance=1.08, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(5, 0, resistance=40, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(5, 6, resistance=2.01, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(5, 8, resistance=1.25, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(6, 9, resistance=2.44, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(7, 0, resistance=40, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(7, 8, resistance=1.79, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(8, 9, resistance=2.01, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(9, 0, resistance=44.1, voltage=0, type='СИП', length=1000, I=0)
    return graph

# stop source data

graph = func_initialization(directed_adjacency_list, COUNT_NODES, COUNT_BRANCHES)

zero_potential = COUNT_NODES - 1

conductivity_matrix = np.zeros((COUNT_NODES-1, COUNT_NODES-1))
current_matrix = np.zeros((COUNT_NODES-1, 1))
export_array = []
import_array = []
for potential in range(COUNT_NODES): # за данный проход формируется уравнение узловых потенциалов относительно рассматриваемого узла
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
        if (index_matrix == zero_potential):
            continue
        if (potential == index_matrix):
            if (len(export_array) == 1):
                conductivity_matrix[potential][index_matrix] += 1/(graph[potential][export_array[0]]['resistance'])
                #current_matrix[potential] -= (graph[potential][export_array[0]]['voltage']) / (graph[potential][export_array[0]]['resistance'])
            else:
                for exp_arr in range(len(export_array)):
                    conductivity_matrix[potential][index_matrix] += 1 / (graph[potential][export_array[exp_arr]]['resistance'])
                    #current_matrix[potential] -= (graph[potential][export_array[exp_arr]]['voltage']) / (graph[potential][export_array[exp_arr]]['resistance'])
            if (len(import_array) == 1):
                conductivity_matrix[potential][index_matrix] += 1 / (graph[import_array[0]][potential]['resistance'])
                #current_matrix[potential] += (graph[import_array[0]][potential]['voltage']) / (graph[import_array[0]][potential]['resistance'])
            else:
                for imp_arr in range(len(import_array)):
                    conductivity_matrix[potential][index_matrix] += 1 / (graph[import_array[imp_arr]][potential]['resistance'])
                    #current_matrix[potential] += (graph[import_array[imp_arr]][potential]['voltage']) / (graph[import_array[imp_arr]][potential]['resistance'])
    if (len(export_array) == 1):
        if (export_array[0] != zero_potential):
            conductivity_matrix[potential][export_array[0]] -= 1 / (graph[potential][export_array[0]]['resistance'])
            current_matrix[potential] -= (graph[potential][export_array[0]]['voltage']) / (graph[potential][export_array[0]]['resistance'])
    else:
        for exp_arr in range(len(export_array)):
            if (export_array[exp_arr] == zero_potential):
                continue
            conductivity_matrix[potential][export_array[exp_arr]] -= 1 / (graph[potential][export_array[exp_arr]]['resistance'])
            current_matrix[potential] -= (graph[potential][export_array[exp_arr]]['voltage']) / (graph[potential][export_array[exp_arr]]['resistance'])
    if (len(import_array) == 1):
        if (import_array[0] != zero_potential):
            conductivity_matrix[potential][import_array[0]] -= 1 / (graph[import_array[0]][potential]['resistance'])
            current_matrix[potential] += (graph[import_array[0]][potential]['voltage']) / (graph[import_array[0]][potential]['resistance'])
    else:
        for imp_arr in range(len(import_array)):
            if (import_array[imp_arr] == zero_potential):
                continue
            conductivity_matrix[potential][import_array[imp_arr]] -= 1 / (graph[import_array[imp_arr]][potential]['resistance'])
            current_matrix[potential] += (graph[import_array[imp_arr]][potential]['voltage']) / (graph[import_array[imp_arr]][potential]['resistance'])

    export_array.clear()
    import_array.clear()

#print(conductivity_matrix)
#print(current_matrix)
potential_matrix = np.linalg.solve(conductivity_matrix, current_matrix)

for nodes in range(len(potential_matrix)):
    graph.nodes[nodes]['potential'] = float(potential_matrix[nodes])

for branch in graph.edges():
    graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] - graph.nodes[branch[1]]['potential'] + graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]]['resistance']
    #print("I (", index, ") = ", graph[branch[0]][branch[1]]['I'])

for branch in graph.edges():
    print(graph.edges[branch])