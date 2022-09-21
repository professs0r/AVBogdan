import numpy as np
import networkx as nx
import operator
import math

#GLOBAL VARIABLES
x_0 = 0.336                                                                # for AL (D = 600 mm) cross_section = 35 mm^2
resistance_Al = 0.028                                                                                       # Ohm*mm^2/m
resistance_Cu = 0.0175                                                                                      # Ohm*mm^2/m
k_form = 4/3                                                                                          # form coefficient
k_z = 0.5                                                                                    # coefficient complete form
k_konf = 0.99                                                      # coefficient configuration active and reactive power
delta_p_l_y_6 = 0.0000308                 # кВт*ч/(м*ч) удельные годовые потери ЭЭ от токов утечки по изоляторам ВЛ-6 кВ
delta_p_l_y_10 = 0.0000502                # кВт*ч/(м*ч) удельные годовые потери ЭЭ от токов утечки по изоляторам ВЛ-6 кВ
tg_y_k = 0.33           # необходимое значение коэффициента реактивной мощности после установки конденсаторной установки
cos_y_k = 0.95                                                              # необходимое значение коэффициента мощности

# потом удалить
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
        graph.add_node(index, potential=0, I=0.0)
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

graph = func_initialization(directed_adjacency_list, 10, 17)
# потом удалить


# start functions and support elements

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

def func_directed_to_nondirected_adjacency_list(directed_adjacency_list):
    """
    функция, которая на основе списка (ориентированного) смежности графа, возвращает список смежности неориентиро-
    ванного графа
    :param directed_adjacency_list:
    :return:
    WORKING CORRECT!!
    """
    nondirected_adjacency_list = directed_adjacency_list.copy()
    for nodes in range(len(directed_adjacency_list)):
        temp_tuple = ()
        for points in range(len(directed_adjacency_list)):
            if nodes == points:
                continue
            else:
                if (isinstance(directed_adjacency_list[points], int)):
                    if nodes == directed_adjacency_list[points]:
                        temp_tuple += (int(points),)
                    continue
                if nodes in directed_adjacency_list[points]:
                    temp_tuple += (int(points), )
        if (isinstance(directed_adjacency_list[nodes], int)):
            temp_tuple += (directed_adjacency_list[nodes],)
            nondirected_adjacency_list[nodes] = tuple(sorted(temp_tuple))
        else:
            if (isinstance(temp_tuple, int)):
                nondirected_adjacency_list[nodes] += temp_tuple
                nondirected_adjacency_list[nodes] = tuple(sorted(nondirected_adjacency_list[nodes]))
            else:
                nondirected_adjacency_list[nodes] += temp_tuple
                nondirected_adjacency_list[nodes] = tuple(sorted(nondirected_adjacency_list[nodes]))
    return nondirected_adjacency_list

def func_initialization_undirected_adjacency_list(edges):                    # ВОЗМОЖНО, НУЖНО БУДЕТ УДАЛИТЬ ЭТУ ФУНКЦИЮ
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

def func_list_to_matrix(adjacency_list, nodes):
    """
    функция из списка смежности возвращает матрицу смежности
    :param adjacency_list: список смежности
    :return: матрица смежности
    WORKING CORRECT!
    """
    adjacency_matrix = np.zeros((nodes, nodes))
    for row in range(len(adjacency_list)):
        if(isinstance(adjacency_list[row], int)):
            adjacency_matrix[row][adjacency_list[row]] = 1
        else:
            for column in range(len(adjacency_list[row])):
                adjacency_matrix[row][adjacency_list[row][column]] = 1
    return adjacency_matrix

def func_dict_to_directed_matrix(dictionary_of_tree):
    """

    :param dictionary_of_tree: dict of spanning tree
    :return: adjacency matrix
    """
    adjacency_matrix = np.zeros((len(dictionary_of_tree)+1, len(dictionary_of_tree)+1))
    for item in dictionary_of_tree.items():
        adjacency_matrix[item[1]][item[0]] = 1
    return adjacency_matrix

def func_dict_to_undirected_matrix(dictionary_of_tree):
    """

    :param dictionary_of_tree: dict of spanning tree
    :return: adjacency matrix
    """
    adjacency_matrix = np.zeros((len(dictionary_of_tree)+1, len(dictionary_of_tree)+1))
    for item in dictionary_of_tree.items():
        adjacency_matrix[item[1]][item[0]] = 1
        adjacency_matrix[item[0]][item[1]] = 1
    return adjacency_matrix

def func_dict_to_graph(dictionary_of_tree, edges, flag=1):
    """

    :param dictionary: dict of spanning tree
    :edges: edges of graph with supporting values
    :flag: 1 - is directed graph, 0 - undirected graph (default vaule = 1, so directed graph)
    :return: graph
    FIX ME!
    """
    graph = nx.DiGraph()
    for index in dictionary_of_tree.keys():
        graph.add_node(index, potential=0, active=15, I=0)
    if flag == 1:
        dictionary_of_tree.pop(int(list(dictionary_of_tree.keys())[0]))
        temp_edges = edges.copy()
        for finish, start in dictionary_of_tree.items():
            for iter in range(len(temp_edges)):
                if int(temp_edges[iter][0]) == start and int(temp_edges[iter][1]) == finish:
                    graph.add_edge(int(temp_edges[iter][0]), int(temp_edges[iter][1]), resistance=float(temp_edges[iter][2]),
                                   voltage=float(temp_edges[iter][3]), type=temp_edges[iter][4], length=float(temp_edges[iter][5]),
                                   cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]), material=temp_edges[iter][8],
                                   r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]), cos_y=float(temp_edges[iter][11]),
                                   sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                                   lose_energy=float(temp_edges[iter][14]))
                    temp_edges = np.delete(temp_edges, iter, axis=0)
                    break
    else:
        dictionary_of_tree.pop(next(dictionary_of_tree.__iter__()))
        temp_edges = edges.copy()
        for finish, start in dictionary_of_tree.items():
            for iter in range(len(temp_edges)):
                if (int(temp_edges[iter][0]) == start or int(temp_edges[iter][0]) == finish)\
                        and\
                   (int(temp_edges[iter][1]) == finish or int(temp_edges[iter][1]) == start):
                    graph.add_edge(int(temp_edges[iter][0]), int(temp_edges[iter][1]), resistance=float(temp_edges[iter][2]),
                                   voltage=float(temp_edges[iter][3]), type=temp_edges[iter][4], length=float(temp_edges[iter][5]),
                                   cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]),
                                   material=temp_edges[iter][8],
                                   r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]), cos_y=float(temp_edges[iter][11]),
                                   sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                                   lose_energy=float(temp_edges[iter][14]))
                    graph.add_edge(int(temp_edges[iter][1]), int(temp_edges[iter][0]), resistance=float(temp_edges[iter][2]),
                                   voltage=float(temp_edges[iter][3]), type=temp_edges[iter][4], length=float(temp_edges[iter][5]),
                                   cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]),
                                   material=temp_edges[iter][8],
                                   r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]), cos_y=float(temp_edges[iter][11]),
                                   sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                                   lose_energy=float(temp_edges[iter][14]))
                    temp_edges = np.delete(temp_edges, iter, axis=0)
                    break
    return graph

def func_edges_to_directed_graph(edges, count_nodes):
    """
    функция для инициализации ориентированного графа на основе списка рёбер
    :param edges: список рёбер
    :return: ориентированный граф
    """
    graph = nx.DiGraph()
    for index in range(count_nodes):
        graph.add_node(index, potential=0.0, active=15.0, I=0.0, root=None, parent=None, visited=False)
    temp_edges = edges.copy()
    for iter in range(len(edges)):
        graph.add_edge(int(temp_edges[iter][0]), int(temp_edges[iter][1]), resistance=float(temp_edges[iter][2]),
                       voltage=float(temp_edges[iter][3]), type=int(temp_edges[iter][4]), length=float(temp_edges[iter][5]),
                       cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]), material=temp_edges[iter][8],
                       r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]), cos_y=float(temp_edges[iter][11]),
                       sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                       lose_energy=float(temp_edges[iter][14]))
    return graph

def func_edges_to_undirected_graph(edges, count_nodes):
    """
    функция для инициализации неориентированного графа на основе списка рёбер
    :param edges:
    :param count_nodes:
    :return:
    """
    graph = nx.Graph()
    """
    # active load in each node 15 kW
    for index in range(count_nodes):
        graph.add_node(index, potential=0.0, active=15.0, I=0.0, root=None, parent=None, visited=False, weight=index+1)
    """
    # power in kW
    """
    graph.add_node(0, potential=0.0, active=0, I=0.0, root=None, parent=None, visited=False, weight=1)
    graph.add_node(1, potential=0.0, active=10, I=0.0, root=None, parent=None, visited=False, weight=2)
    graph.add_node(2, potential=0.0, active=15, I=0.0, root=None, parent=None, visited=False, weight=3)
    graph.add_node(3, potential=0.0, active=2.3, I=0.0, root=None, parent=None, visited=False, weight=4)
    graph.add_node(4, potential=0.0, active=5, I=0.0, root=None, parent=None, visited=False, weight=5)
    graph.add_node(5, potential=0.0, active=4, I=0.0, root=None, parent=None, visited=False, weight=6)
    graph.add_node(6, potential=0.0, active=2.5, I=0.0, root=None, parent=None, visited=False, weight=7)
    graph.add_node(7, potential=0.0, active=4, I=0.0, root=None, parent=None, visited=False, weight=8)
    graph.add_node(8, potential=0.0, active=1, I=0.0, root=None, parent=None, visited=False, weight=9)
    graph.add_node(9, potential=0.0, active=3.6, I=0.0, root=None, parent=None, visited=False, weight=10)
    """
    for index in range(count_nodes):
        graph.add_node(index, potential=0.0, active=15.0, I=0.0, root=None, parent=None, visited=False)
    temp_edges = edges.copy()
    for iter in range(len(edges)):
        graph.add_edge(int(temp_edges[iter][0]), int(temp_edges[iter][1]), resistance=float(temp_edges[iter][2]),
                       voltage=float(temp_edges[iter][3]), type=int(temp_edges[iter][4]),
                       length=float(temp_edges[iter][5]),
                       cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]),
                       material=temp_edges[iter][8],
                       r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]),
                       cos_y=float(temp_edges[iter][11]),
                       sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                       lose_energy=float(temp_edges[iter][14]))
    return graph

def func_make_matrix_incidence(graph):
    """
    функция для составления матрицы инцидентности графа
    :param graph:
    :return:
    """
    count_nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    matrix_incidence = np.zeros((count_nodes, edges))
    iter = 0
    for branch in graph.edges.items():
        matrix_incidence[int(branch[0][0])][iter] = 1
        matrix_incidence[int(branch[0][1])][iter] = -1
        iter += 1
    return matrix_incidence


def func_initialization(nodes):
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
        graph.add_node(index, potential=0, active=15, I=0)
    graph.add_edge(0, 1, resistance=70.1, voltage=630, type='СИП', length=701, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(0, 3, resistance=5.62, voltage=220, type='СИП', length=56.2, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(1, 2, resistance=2.55, voltage=0, type='СИП', length=25.5, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(1, 4, resistance=70, voltage=0, type='СИП', length=700, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(2, 3, resistance=85.89, voltage=0, type='СИП', length=858.9, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(2, 6, resistance=3.69, voltage=0, type='СИП', length=36.9, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(3, 6, resistance=2.33, voltage=0, type='СИП', length=23.3, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(4, 0, resistance=1.52, voltage=0, type='СИП', length=15.2, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(5, 1, resistance=1.35, voltage=380, type='СИП', length=13.5, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(5, 4, resistance=0.1, voltage=0, type='СИП', length=1, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    graph.add_edge(6, 5, resistance=0.84, voltage=0, type='СИП', length=8.4, cross_section=35, I=0, material='Al', r_0=0, x_0=0, cos_y=0.89, sin_y=0, lose_volt=0, lose_energy=0)
    return graph

def func_initialization_undirected_graph(list, nodes):
    """
    функция, которая на основе списка смежности, а также сведений о количестве узлов и ветвей создаёт неориентированный
    граф
    :param list: список смежности графа
    :param nodes: количество вершин графа (узлов схемы)
    :param branches: количество рёбер графа (ветвей схемы)
    :return: неориентированный граф
    FIX ME!!
    """
    graph = nx.Graph()
    for index in range(nodes):
        graph.add_node(index, potential=0, active=15, I=0)
    for node in range(len(list)):
        for point in list[node]:
            graph.add_edge(int(node), int(point), resistance=0,
                           voltage=0, type=0, length=0,
                           cross_section=0, I=0, material=0,
                           r_0=0, x_0=0, cos_y=0,
                           sin_y=0, lose_volt=0,
                           lose_energy=0)
    return graph

def func_initialization_directed_graph(list, nodes):
    """
    функция, которая на основе списка смежности, а также сведений о количестве узлов и ветвей создаёт ориентированный
    граф
    :param list: список рёбер графа, с дополнительной информацией
    :param nodes: количество вершин графа (узлов схемы)
    :param branches: количество рёбер графа (ветвей схемы)
    :return: ориентированный граф
    """
    graph= nx.DiGraph()
    for index in range(nodes):
        graph.add_node(index, potential=0, active=15, I=0)
    for node in range(len(list)):
        if (isinstance(list[node], int)):
            graph.add_edge(int(node), int(list[node]), resistance=0,
                           voltage=0, type=0, length=0,
                           cross_section=0, I=0, material=0,
                           r_0=0, x_0=0, cos_y=0,
                           sin_y=0, lose_volt=0,
                           lose_energy=0)
        else:
            for point in list[node]:
                graph.add_edge(int(node), int(point), resistance=0,
                               voltage=0, type=0, length=0,
                               cross_section=0, I=0, material=0,
                               r_0=0, x_0=0, cos_y=0,
                               sin_y=0, lose_volt=0,
                               lose_energy=0)
    return graph

def func_count_of_nodes(adjacency_matrix):
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

def func_count_of_branches(adjacency_list):
    """
    функция по списку смежности считает количество ветвей (рёбер) в графе
    :param adjacency_list:
    :return: count_branches
    WORKING CORRECT!
    """
    branches = 0
    for elements in range(len(adjacency_list)):
        if (isinstance(adjacency_list[elements], int)):
            branches += 1
            continue
        branches += int(len(adjacency_list[elements]))
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

def func_BFS(graph, start):
    """
    обход графа в ширину
    :param graph: граф
    :param start: стартовая вершина
    :return:
    WORKING CORRECT!!
    """
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
    FIX ME!!
    """
    matrix_Kirchoff = adjacency_matrix.copy()
    matrix_Kirchoff *= -1
    for diagonal in range(len(matrix_Kirchoff)):
        temp_sum = sum(matrix_Kirchoff)
        matrix_Kirchoff[diagonal][diagonal] = -1 * sum(matrix_Kirchoff[diagonal])
    temp1 = np.delete(matrix_Kirchoff, len(matrix_Kirchoff) - 1, 0)
    temp2 = np.delete(temp1, len(matrix_Kirchoff) - 1, 1)
    print("Количество остовных деревьев = ", np.linalg.det(temp2))
    return np.linalg.det(temp2)

def func_BFS_for_spanning_trees(graph, node, visited, path, count_path, trees):
    """

    :param graph:
    :param node:
    :param visited:
    :param path:
    :param trees:
    :return:
    """
    to_explore = [node]
    visited.add(node)
    path[node] = node
    while to_explore:
        next = to_explore.pop(0)
        new_vertexes = [i for i in graph[next] if i not in visited]
        for i in new_vertexes:
            path[i] = next
            count_path += 1
            func_DFS_for_spanning_trees(graph, i, visited, path, count_path, trees)
        to_explore.extend(new_vertexes)
        visited.update(new_vertexes)

def func_DFS_for_spanning_trees(graph, node, visited, path, count_path, trees):
    """
    function for support "func_spanning_trees" function
    :param graph:
    :param node:
    :param visited:
    :return:
    FIX ME!
    """
    if node in visited:
        return
    visited.add(node)
    if count_path == len(graph.nodes):
        sort_path = sorted(path)
        path_to_memory = path.copy()
        if (np.array(sort_path) == np.array(graph.nodes)).all() and (path_to_memory not in trees):
            trees.append(path_to_memory)
    for neighbour in graph[node]:
        if neighbour not in visited:
            path[neighbour] = node
            count_path += 1
            func_DFS_for_spanning_trees(graph, neighbour, visited, path, count_path, trees)
            visited.remove(neighbour)
            path.pop(neighbour)
            count_path -= 1

def func_spanning_trees(graph):
    """
    count and output spanning trees in graph
    :param graph:
    :return:
    """
    visited = set()
    trees = []
    path = dict()
    count_path = 1
    for start_point in graph.nodes:
        func_BFS_for_spanning_trees(graph, start_point, visited, path, count_path, trees)
        visited.clear()
        path.clear()
        count_path = 1
    print("Количество остовных деревьев = ", len(trees))
    for iter in range(len(trees)):
        print("Остовное дерево ", iter + 1, ": ", trees[iter])
    return trees

def func_calculating_support_variables(graph):
    """
    function for calculating support variables for forward work
    :param graph:
    :return:
    FIX ME!!
    """
    for g in graph.edges():
        if graph[g[0]][g[1]]['material'] == 'Al':
            graph[g[0]][g[1]]['r_0'] = resistance_Al/graph[g[0]][g[1]]['cross_section']
        else:
            print(graph[g[0]][g[1]]['cross_section'])
            graph[g[0]][g[1]]['r_0'] = resistance_Cu / graph[g[0]][g[1]]['cross_section']
        graph[g[0]][g[1]]['x_0'] = 0.000336
        graph[g[0]][g[1]]['sin_y'] = math.sqrt(pow(1, 2) - pow(graph[g[0]][g[1]]['cos_y'], 2))
        # при необходимости, строку ниже можно закомментировать, чтобы использовать заданные значения
        #graph[g[0]][g[1]]['resistance'] = graph[g[0]][g[1]]['r_0'] * graph[g[0]][g[1]]['length']

def func_loses_voltage(graph):
    """
    function which calculate loses voltage in directed graph
    :param graph: directed graph
    :return:
    """
    print("func_loses_voltage")
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['lose_volt'] = math.sqrt(3) * graph[edge[0]][edge[1]]['I'] * graph[edge[0]][edge[1]]['length'] * \
                            (graph[edge[0]][edge[1]]['r_0'] * graph[edge[0]][edge[1]]['cos_y'] +
                             graph[edge[0]][edge[1]]['x_0'] * graph[edge[0]][edge[1]]['sin_y'])
        print(graph[edge[0]][edge[1]]['lose_volt'])

def func_loses_energy_high(graph):
    """
    function which calculate loses energy in directed graph (for voltage 1000 V and high)
    :param graph:
    :return:
    """
    print("func_loses_energy_high_voltage")
    # табличная величина зависит от сечения проводника, типа проводника (ВЛ или КЛ) и напряжения
    delta_lose = 0
    # примем расчётное время за величину в 1 час
    T_p = 1
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['lose_energy'] = ((3*k_konf*k_form*T_p*graph[edge[0]][edge[1]]['r_0']*
                                                  graph[edge[0]][edge[1]]['length']*
                                                  pow(graph[edge[0]][edge[1]]['I']), 2)/(1000)) + (delta_lose*
                                                  graph[edge[0]][edge[1]]['length']*T_p)
        """
        вариант с величиной потребляемой энергии по приборам учёта (вместо тока)
        graph[edge[0]][edge[1]]['lose_energy'] = ((k_konf*k_form*pow(graph[edge[0]][edge[1]]['energy'], 2)*
                                                  graph[edge[0]][edge[1]]['r_0']
                                                  graph[edge[0]][edge[1]]['length'])/(1000*T_p*
                                                  pow((0.4*graph[edge[0]][edge[1]]['cos_y']), 2))) + (delta_lose*
                                                  graph[edge[0]][edge[1]]['length']*T_p)
        """
        print(graph[edge[0]][edge[1]]['lose_energy'])

def func_loses_energy_400(graph):
    """
    function which calculate loses energy in directed graph (for voltage 400 V)
    :param graph:
    :return:
    """
    print("func_loses_energy_400")
    # примем расчётное время за величину в 1 час
    T_p = 1
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['lose_energy'] = (3*k_konf*k_form*T_p*graph[edge[0]][edge[1]]['r_0']*
                                                  graph[edge[0]][edge[1]]['length']*
                                                  pow(graph[edge[0]][edge[1]]['I'], 2))/(1000)
        """
        вариант с величиной потребляемой энергии по приборам учёта (вместо тока)
        graph[edge[0]][edge[1]]['lose_energy'] = (k_konf*k_form*pow(graph[edge[0]][edge[1]]['energy'], 2)*
                                                  graph[edge[0]][edge[1]]['r_0']
                                                  graph[edge[0]][edge[1]]['length'])/(1000*T_p*
                                                  pow((0.4*graph[edge[0]][edge[1]]['cos_y']), 2))
        """
        print(graph[edge[0]][edge[1]]['lose_energy'])

def func_loses_energy_220(graph):
    """
    function which calculate loses energy in directed graph (for voltage 220 V)
    :param graph:
    :return:
    """
    print("func_loses_energy_220")
    # примем расчётное время за величину в 1 час
    T_p = 1
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['lose_energy'] = 2*(k_konf*k_form*T_p*graph[edge[0]][edge[1]]['r_0']*
                                                    graph[edge[0]][edge[1]]['length']*
                                                    pow(graph[edge[0]][edge[1]]['I']), 2)/(1000)
        """
        вариант с величиной потребляемой энергии по приборам учёта (вместо тока)
        graph[edge[0]][edge[1]]['lose_energy'] = 2*(k_konf*k_form*pow(graph[edge[0]][edge[1]]['energy'], 2)*
                                                    graph[edge[0]][edge[1]]['r_0']
                                                    graph[edge[0]][edge[1]]['length'])/(1000*T_p*
                                                    pow((0.4*graph[edge[0]][edge[1]]['cos_y']), 2))
        """
        print(graph[edge[0]][edge[1]]['lose_energy'])

def func_algo_AVBogdan(graph):
    """
    function to count and find spanning trees in graph by A.V. Bogdan
    :param graph:
    :return:
    """
    matrix_AV = np.zeros((len(graph.nodes), len(graph.edges)))
    matrix_incidence =np.zeros((len(graph.nodes), len(graph.edges)))
    iter = 0
    for branch in graph.edges:
        matrix_AV[branch[0]][iter] = 1
        matrix_AV[branch[1]][iter] = -1
        matrix_incidence[branch[0]][iter] = 1
        matrix_incidence[branch[1]][iter] = 1
        iter += 1
    print("Матрица Александра Владимировича: ")
    print(matrix_AV)
    print("Матрица инцидентности исходной схемы: ")
    print(matrix_incidence)
    """
    print("После удаления последней строки: ")
    matrix_AV = np.delete(matrix_AV, len(matrix_AV) - 1, 0)
    print(matrix_AV)
    """

def func_calculated_reactive_compens(graph):
    """
    calculated compensation reactive power
    :param graph:
    :return:
    FIX ME!!!!
    """
    # линейное напряжение принимаем равным в 400 В, однако, я думаю, что лучше прописать соответствующее поле в графе
    U_l = 400
    for edge in graph.edges():
        S = graph[edge[0]][edge[1]]['I']*U_l
        tg_y = math.sqrt((1/pow(graph[edge[0]][edge[1]]['cos_y'], 2)) - 1)
        Q_ku = math.sqrt(3)*graph[edge[0]][edge[1]]['I']*U_l*graph[edge[0]][edge[1]]['cos_y']*(tg_y - tg_y_k)


def func_calculated_current_node_potential_algo(graph):
    """
    method node potential
    :return:
    """
    count_nodes = int(graph.number_of_nodes())
    count_branches = int(graph.number_of_edges())
    zero_potential = int(count_nodes - 1)
    conductivity_matrix = np.zeros((count_nodes - 1, count_nodes - 1))
    current_matrix = np.zeros((count_nodes - 1, 1))
    export_array = []
    import_array = []
    matrix_incidence = func_make_matrix_incidence(graph)
    list_edges = []
    for edge in graph.edges.items():
        list_edges.append(edge[0])
    for potential in range(count_nodes):  # за данный проход формируется уравнение узловых потенциалов относительно рассматриваемого узла
        if (potential == zero_potential):
            continue
        for edge in range(count_branches):
            if matrix_incidence[potential][edge] == 1:
                export_array.append(list_edges[edge][1])
            elif matrix_incidence[potential][edge] == -1:
                import_array.append(list_edges[edge][0])
        for index_matrix in range(len(conductivity_matrix[potential])):
            if (index_matrix == zero_potential):
                continue
            if (potential == index_matrix):
                if (len(export_array) == 1):
                    conductivity_matrix[potential][index_matrix] += 1 / (
                    graph[potential][export_array[0]]['resistance'])
                elif (len(export_array) > 1):
                    for exp_arr in range(len(export_array)):
                        conductivity_matrix[potential][index_matrix] += 1 / (graph[potential][export_array[exp_arr]]['resistance'])
                if (len(import_array) == 1):
                    conductivity_matrix[potential][index_matrix] += 1 / (
                    graph[import_array[0]][potential]['resistance'])
                elif (len(import_array) > 1):
                    for imp_arr in range(len(import_array)):
                        conductivity_matrix[potential][index_matrix] += 1 / (
                        graph[import_array[imp_arr]][potential]['resistance'])
        if (len(export_array) == 1):
            if (export_array[0] != zero_potential):
                conductivity_matrix[potential][export_array[0]] -= 1 / (graph[potential][export_array[0]]['resistance'])
                current_matrix[potential] -= (graph[potential][export_array[0]]['voltage']) / (
                graph[potential][export_array[0]]['resistance'])
        elif (len(export_array) > 1):
            for exp_arr in range(len(export_array)):
                if (export_array[exp_arr] == zero_potential):
                    continue
                conductivity_matrix[potential][export_array[exp_arr]] -= 1 / (
                graph[potential][export_array[exp_arr]]['resistance'])
                current_matrix[potential] -= (graph[potential][export_array[exp_arr]]['voltage']) / (
                graph[potential][export_array[exp_arr]]['resistance'])
        if (len(import_array) == 1):
            if (import_array[0] != zero_potential):
                conductivity_matrix[potential][import_array[0]] -= 1 / (graph[import_array[0]][potential]['resistance'])
                current_matrix[potential] += (graph[import_array[0]][potential]['voltage']) / (
                graph[import_array[0]][potential]['resistance'])
        elif (len(import_array) > 1):
            for imp_arr in range(len(import_array)):
                if (import_array[imp_arr] == zero_potential):
                    continue
                conductivity_matrix[potential][import_array[imp_arr]] -= 1 / (
                graph[import_array[imp_arr]][potential]['resistance'])
                current_matrix[potential] += (graph[import_array[imp_arr]][potential]['voltage']) / (
                graph[import_array[imp_arr]][potential]['resistance'])

        export_array.clear()
        import_array.clear()

    potential_matrix = np.linalg.solve(conductivity_matrix, current_matrix)

    for nodes in range(len(potential_matrix)):
        graph.nodes[nodes]['potential'] = float(potential_matrix[nodes])

    for branch in graph.edges():
        graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] - graph.nodes[branch[1]]['potential'] +
                                            graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]][
                                               'resistance']

# end functions and support elements

# потом удалить
#func_calculated_current_node_potential_algo(graph)
# потом удалить