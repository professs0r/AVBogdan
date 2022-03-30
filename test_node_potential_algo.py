import numpy as np
import networkx as nx
import operator
from collections import deque

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

# stop source data

# start functions and support elements

directed_adjacency_list = np.array([(1, 3),
                                    (2, 4),
                                    (3, 6),
                                    (6),
                                    (0),
                                    (1, 4),
                                    (5)])

def func_initialization_adjacency_list(edges):
    """
    функция, которая на основе списка рёбер создаёт список смежности (первичный [исходный])
    :param edges: список рёбер (каждый элемент списка содержит в себе сведения о ребер: длина, ток, наяпржение)
    :return: список смежности
    WORKING CORRECT!
    """
    adjacency_list = np.empty((len(edges), 1))
    temp_array_edges = []
    for all_edges in range(len(edges)):
        temp_tuple = (int(edges[all_edges][0]), int(edges[all_edges][1]))
        temp_array_edges.append(temp_tuple)
    max_num_of_node = max(temp_array_edges, key=operator.itemgetter(0))
    temp_array_adjacency = []
    for element in range(max(max_num_of_node) + 1):
        temp_tuple = ()
        for index in range(len(temp_array_edges)):
            if element == temp_array_edges[index][0]:
                temp_tuple += (temp_array_edges[index][1], )
        temp_array_adjacency.append(temp_tuple)
    adjacency_list = np.asarray(temp_array_adjacency)
    return adjacency_list

def func_initialization_undirected_adjacency_list(edges): # ВОЗМОЖНО, НУЖНО БУДЕТ УДАЛИТЬ ЭТУ ФУНКЦИЮ
    """
    функция, которая на основе списка рёбер (ветвей) создаёт неориентированный список смежности (первичный [исходный])
    :param edges: список рёбер (ветвей схемы)
    :return: список смежности для неориентированного графа
    WORKING CORRECT!
    """
    undirected_adjacency_list = np.empty((len(edges), 1))
    max_num_of_node = int(np.amax(edges, axis=0)[0])
    temp_array_undirected_adjacency = []
    for node in range(max_num_of_node + 1):
        temp_tuple = ()
        for index in range(len(edges)):
            if node == edges[index][0]:
                temp_tuple += (int(edges[index][1]), )
            if node == edges[index][1]:
                temp_tuple += (int(edges[index][0]), )
        temp_array_undirected_adjacency.append(temp_tuple)
    undirected_adjacency_list = np.asarray(temp_array_undirected_adjacency)
    return undirected_adjacency_list

def func_list_to_matrix(adjacency_list):
    """
    функция из списка смежности возвращает матрицу смежности
    :param adjacency_list: список смежности
    :return: матрица смежности
    WORKING CORRECT!
    """
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)))
    for row in range(len(adjacency_list)):
        if(len(adjacency_list[row]) == 1):
            adjacency_matrix[row][adjacency_list[row]] = 1
        else:
            for column in range(len(adjacency_list[row])):
                adjacency_matrix[row][adjacency_list[row][column]] = 1

    return adjacency_matrix

def func_initialization(list, nodes, branches):
    """
    функция инициализации ориентированного графа
    :param list: список смежности
    :param nodes: количество узлов
    :param branches: количество ветвей
    :return: ориентированный граф
    WORKING CORRECT!
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

def func_initialization_undirected_graph(list, nodes, branches):
    """
    функция, которая на основе списка смежности, а также сведений о количестве узлов и ветвей создаёт неориентированный
    граф
    :param list: список смежности графа
    :param nodes: количество вершин графа (узлов схемы)
    :param branches: количество рёбер графа (ветвей схемы)
    :return: неориентированный граф
    WORKING CORRECT!
    """
    graph= nx.Graph()
    for index in range(nodes):
        graph.add_node(index, potential=0)
    for branch in range(branches):
        graph.add_edge(list[branch][0], list[branch][1], resistance=list[branch][2], voltage=list[branch][3],
                       type=list[branch][4], length=list[branch][5], I=list[branch][6])
    return graph

def func_initialization_v2(list, nodes, branches):
    """
    функция инициализации ориентированного графа
    :param list: список смежности
    :param nodes: количество узлов в графе
    :param branches: количество ветвей в графе
    :return: ориентированный граф
    WORKING CORRECT!
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
    WORKING CORRECT!
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
    WORKING CORRECT!
    """
    branches = 0
    for elements in range(adjacency_list.size):
        if (isinstance(adjacency_list[elements], int)):
            branches += 1
            continue
        branches += len(adjacency_list[elements])
    return branches

def func_count_of_undirected_branches(directed_adjacency_matrix):
    """
    функция подсчёта количества рёбер в неориентированном графе
    :param directed_adjacency_matrix: матрица смежности для неориентированного графа
    :return: количество ветвей
    WORKING!!!
    """
    index = 0
    count_of_branches = 0
    for index in range(directed_adjacency_matrix[0].size):
        iterator = index
        while iterator < directed_adjacency_matrix[0].size:
            if directed_adjacency_matrix[index][iterator] == 1:
                count_of_branches += 1
            iterator += 1
    return count_of_branches

def func_DFS(graph, node, visited):
    """
    обход графа в глубину
    :param graph: неориентированный граф
    :param node: рассматриваемый узел
    :param visited: список уже посещённых узлов
    :return:
    """
    if node in visited:
        return
    visited.add(node)
    for neighbour in graph[node]:
        if neighbour not in visited:
            func_DFS(graph, neighbour, visited)

def func_run_DFS(graph):
    """
    функция, которая запускает функцию обхода графа в глубину через цикл
    :param graph: неориентированный граф
    :return:
    """
    visited = set()
    N = 0
    for node in graph:
        if node not in visited:
            func_DFS(graph, node, visited)
            N += 1
    print(visited)
    print("Количество компонент связности = ", N)

def func_BFS(graph):
    """
    обход графа в ширину
    :param graph: граф
    :return:
    """
    start = 0
    visited = {start}
    to_explore = [start]
    lens = dict()
    lens[start] = 0
    prev = dict()
    prev[start] = start
    while to_explore:
        next = to_explore.pop(0)
        new_vertexes = [i for i in graph[next] if i not in visited]
        for i in new_vertexes:
            lens[i] = lens[next] + 1
            prev[i] = next
        to_explore.extend(new_vertexes)
        visited.update(new_vertexes)
    return lens, prev

def func_Kirchhoff(adjacency_matrix):
    """
    матричная теорема о деревьях Кирхгофа
    :param adjacency_matrix: матрица смежности
    :return:
    """
    matrix_Kirchoff = adjacency_matrix.copy()
    matrix_Kirchoff *= -1
    for diagonal in range(len(matrix_Kirchoff)):
        temp_sum = sum(matrix_Kirchoff)
        matrix_Kirchoff[diagonal][diagonal] = -1 * sum(matrix_Kirchoff[diagonal])
    temp1 = np.delete(matrix_Kirchoff, 0, 0)
    temp2 = np.delete(temp1, 0, 1)
    print("Количество остовных деревьев = ", int(np.linalg.det(temp2)))
    return int(np.linalg.det(temp2))


# end functions and support elements

# running algorithm

# teseted
# teseted
# teseted
# teseted

test = func_initialization_undirected_adjacency_list(edges)
test_matrix = func_list_to_matrix(test)
Kirchhoff = func_Kirchhoff(test_matrix)
test_count_nodes = count_of_nodes(test_matrix)
test_count_branches = count_of_branches(test)
test_count_branches_v2 = func_count_of_undirected_branches(test_matrix)
test_graph = func_initialization_undirected_graph(edges, test_count_nodes, 11)
func_run_DFS(test_graph)
print(func_BFS(test_graph))

# teseted
# teseted
# teseted
# teseted
temp_list = func_initialization_adjacency_list(edges)
directed_adjacency_list = temp_list # tested!!!!
directed_adjacency_matrix = func_list_to_matrix(directed_adjacency_list)    # tested!!!!
#directed_adjacency_matrix = func_list_to_matrix(directed_adjacency_list)
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
            else:
                for exp_arr in range(len(export_array)):
                    conductivity_matrix[potential][index_matrix] += 1 / (graph[potential][export_array[exp_arr]]['resistance'])
            if (len(import_array) == 1):
                conductivity_matrix[potential][index_matrix] += 1 / (graph[import_array[0]][potential]['resistance'])
            else:
                for imp_arr in range(len(import_array)):
                    conductivity_matrix[potential][index_matrix] += 1 / (graph[import_array[imp_arr]][potential]['resistance'])
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

potential_matrix = np.linalg.solve(conductivity_matrix, current_matrix)

for nodes in range(len(potential_matrix)):
    graph.nodes[nodes]['potential'] = float(potential_matrix[nodes])

for branch in graph.edges():
    graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] - graph.nodes[branch[1]]['potential'] + graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]]['resistance']

for branch in graph.edges():
    print(graph.edges[branch])
