import numpy as np
import networkx as nx

# start source data

# template of edge
#edge_0 = (source=0, finish=1, resistance=70.1, voltage=630, type='СИП', length=1000, I=0)
edge_0 = (0, 1, 70.1, 630, 0, 1000, 0)
edge_1 = (0, 3, 5.62, 220, 0, 1000, 0)
edge_2 = (1, 2, 2.55, 0, 0, 1000, 0)
edge_3 = (1, 4, 70, 0, 0, 1000, 0)
edge_4 = (2, 3, 85.89, 0, 0, 1000, 0)
edge_5 = (2, 6, 3.69, 0, 0, 1000, 0)
edge_6 = (3, 6, 2.33, 0, 0, 1000, 0)
edge_7 = (4, 0, 1.52, 0, 0, 1000, 0)
edge_8 = (5, 1, 1.35, 380, 0, 1000, 0)
edge_9 = (5, 4, 0.1, 0, 0, 1000, 0)
edge_10 = (6, 5, 0.84, 0, 0, 1000, 0)

edges = np.array([edge_0,
                  edge_1,
                  edge_2,
                  edge_3,
                  edge_4,
                  edge_5,
                  edge_6,
                  edge_7,
                  edge_8,
                  edge_9,
                  edge_10])

directed_adjacency_list = np.array([(1, 3),
                                    (2, 4),
                                    (3, 6),
                                    (6),
                                    (0),
                                    (1, 4),
                                    (5)])
print(type(directed_adjacency_list[0]))

def func_initialization_adjacency_list(edges):
    """
    функция, которая на основе списка рёбер создаёт список смежности (первичный [исходный])
    :param edges: список рёбер (каждый элемент списка содержит в себе сведения о ребер: длина, ток, наяпржение)
    :return: список смежности
    """
    adjacency_list = np.empty((len(edges), 1))
    for all_edges in range(len(edges)):
        temp_tuple = (edges[all_edges][0], edges[all_edges][1])
        # FIX ME
        #np.append(adjacency_list, temp_tuple)
    print(adjacency_list)
    return adjacency_list

def func_list_to_matrix(adjacency_list):
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)))
    for row in range(len(adjacency_list)):
        if (isinstance(adjacency_list[row], int)):
            adjacency_matrix[row][adjacency_list[row]] = 1
        else:
            for column in range(len(adjacency_list[row])):
                adjacency_matrix[row][adjacency_list[row][column]] = 1
    return adjacency_matrix

def func_initialization(list, nodes, branches):
    graph = nx.DiGraph()
    for index in range(nodes):
        graph.add_node(index, potential=0)
    graph.add_edge(0, 1, resistance=70.1, voltage=630, type='СИП', length=1000, I=0)
    graph.add_edge(0, 3, resistance=5.62, voltage=220, type='СИП', length=1000, I=0)
    graph.add_edge(1, 2, resistance=2.55, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(1, 4, resistance=70, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 3, resistance=85.89, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 6, resistance=3.69, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(3, 6, resistance=2.33, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(4, 0, resistance=1.52, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(5, 1, resistance=1.35, voltage=380, type='СИП', length=1000, I=0)
    graph.add_edge(5, 4, resistance=0.1, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(6, 5, resistance=0.84, voltage=0, type='СИП', length=1000, I=0)
    return graph

def func_initialization_v2(list, nodes, branches):
    """
    функция инициализации ориентированного графа
    :param list: список смежности
    :param nodes: количество узлов в графе
    :param branches: количество ветвей в графе
    :return: ориентированный граф
    """
    graph = nx.DiGraph()
    for index in range(nodes):
        graph.add_node(index, potential=0)
    graph.add_edge(0, 1, resistance=70.1, voltage=630, type='СИП', length=1000, I=0)
    graph.add_edge(0, 3, resistance=5.62, voltage=220, type='СИП', length=1000, I=0)
    graph.add_edge(1, 2, resistance=2.55, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(1, 4, resistance=70, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 3, resistance=85.89, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(2, 6, resistance=3.69, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(3, 6, resistance=2.33, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(4, 0, resistance=1.52, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(5, 1, resistance=1.35, voltage=380, type='СИП', length=1000, I=0)
    graph.add_edge(5, 4, resistance=0.1, voltage=0, type='СИП', length=1000, I=0)
    graph.add_edge(6, 5, resistance=0.84, voltage=0, type='СИП', length=1000, I=0)
    return graph

def count_of_nodes(adjacency_matrix):
    """
    функция по матрице смежности считает сколько в графе узлов
    :param adjacency_matrix:
    :return: count_nodes
    """
    count_nodes = 0
    temp_array = np.zeros((len(adjacency_matrix), 1))
    for elements in range(len(adjacency_matrix)):
        temp_array[elements] += sum(adjacency_matrix[elements])
        for accord_elements in range(len(adjacency_matrix)):
            if(accord_elements == elements):
                continue
            temp_array[elements] += adjacency_matrix[accord_elements][elements]
    for index in range(temp_array.size):
        if(temp_array[index] >= 3):
            count_nodes += 1
    return count_nodes

def count_of_branches(adjacency_list):
    """
    функция по списку смежности считает количество ветвей (рёбер) в графе
    :param adjacency_list:
    :return: count_branches
    """
    branches = 0
    for elements in range(adjacency_list.size):
        if (isinstance(adjacency_list[elements], int)):
            branches += 1
            continue
        branches += len(adjacency_list[elements])
    return branches

# stop source data

temp_list = func_initialization_adjacency_list(edges)
directed_adjacency_matrix = func_list_to_matrix(directed_adjacency_list)
count_nodes = count_of_nodes(directed_adjacency_matrix)
count_branches = count_of_branches(directed_adjacency_list)
graph = func_initialization(directed_adjacency_list, count_nodes, count_branches)
graph_v2 = func_initialization_v2(directed_adjacency_list, count_nodes, count_branches)

zero_potential = count_nodes - 1

conductivity_matrix = np.zeros((count_nodes-1, count_nodes-1))
current_matrix = np.zeros((count_nodes-1, 1))
export_array = []
import_array = []
for potential in range(count_nodes): # за данный проход формируется уравнение узловых потенциалов относительно рассматриваемого узла
    if (potential == zero_potential):
        continue
    for node in range(count_nodes):
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