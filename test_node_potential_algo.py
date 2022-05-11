import numpy as np
import networkx as nx
import operator
import math
import matplotlib.pyplot as plt

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

# start source data

# template of edge
#edge_0 = (source=0, finish=1, resistance=70.1, voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated))
edge_0 = (0, 1, 70.1, 630, 0, 701, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_1 = (0, 3, 5.62, 220, 0, 56.2, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_2 = (1, 2, 2.55, 0, 0, 25.5, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_3 = (1, 4, 70, 0, 0, 700, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_4 = (2, 3, 85.89, 0, 0, 858.9, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_5 = (2, 6, 3.69, 0, 0, 36.9, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_6 = (3, 6, 2.33, 0, 0, 23.3, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_7 = (4, 0, 1.52, 0, 0, 15.2, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_8 = (5, 1, 1.35, 380, 0, 13.5, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_9 = (5, 4, 0.1, 0, 0, 1, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_10 = (6, 5, 0.84, 0, 0, 8.4, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)

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

test_list = np.array([(1, 2),
                      (0, 2, 3),
                      (0, 1, 3),
                      (1, 2)])

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

def func_list_to_matrix(adjacency_list):
    """
    функция из списка смежности возвращает матрицу смежности
    :param adjacency_list: список смежности
    :return: матрица смежности
    WORKING CORRECT!
    """
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)))
    for row in range(len(adjacency_list)):
        if(isinstance(adjacency_list[row], int)):
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
        graph.add_node(index, potential=0, active=15)
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
        graph.add_node(index, potential=0, active=15)
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
        graph.add_node(index, potential=0, active=15)
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
    #path[node] = node
    if count_path == len(graph.nodes):
        sort_path = sorted(path)
        if (np.array(sort_path) == np.array(graph.nodes)).all():
            path_to_memory = path.copy()
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

def func_calculated_reactive_compens(graph):
    """
    calculated compensation reactive power
    :param graph:
    :return:
    """
    # линейное напряжение принимаем равным в 400 В, однако, я думаю, что лучше прописать соответствующее поле в графе
    U_l = 400
    for edge in graph.edges():
        S = graph[edge[0]][edge[1]]['I']*U_l
        tg_y = math.sqrt((1/pow(graph[edge[0]][edge[1]]['cos_y'], 2)) - 1)
        Q_ku = math.sqrt(3)*graph[edge[0]][edge[1]]['I']*U_l*graph[edge[0]][edge[1]]['cos_y']*(tg_y - tg_y_k)


def func_calculated_current_node_potential_algo(graph, count_nodes, zero_potential, directed_adjacency_matrix):
    """
    method node potential
    :return:
    """
    print("func_calculated_current_node_potential_algo")
    conductivity_matrix = np.zeros((count_nodes - 1, count_nodes - 1))
    current_matrix = np.zeros((count_nodes - 1, 1))
    export_array = []
    import_array = []
    for potential in range(count_nodes):  # за данный проход формируется уравнение узловых потенциалов относительно рассматриваемого узла
        if (potential == zero_potential):
            continue
        for node in range(count_nodes):
            if (potential == node):
                for export_node in range(len(directed_adjacency_matrix[node])):
                    if (directed_adjacency_matrix[node][export_node] != 0):
                        export_array.append(export_node)
            else:
                if (directed_adjacency_matrix[node][potential] == 1):
                    import_array.append(node)
        for index_matrix in range(len(conductivity_matrix[potential])):
            if (index_matrix == zero_potential):
                continue
            if (potential == index_matrix):
                if (len(export_array) == 1):
                    conductivity_matrix[potential][index_matrix] += 1 / (
                    graph[potential][export_array[0]]['resistance'])
                else:
                    for exp_arr in range(len(export_array)):
                        conductivity_matrix[potential][index_matrix] += 1 / (
                        graph[potential][export_array[exp_arr]]['resistance'])
                if (len(import_array) == 1):
                    conductivity_matrix[potential][index_matrix] += 1 / (
                    graph[import_array[0]][potential]['resistance'])
                else:
                    for imp_arr in range(len(import_array)):
                        conductivity_matrix[potential][index_matrix] += 1 / (
                        graph[import_array[imp_arr]][potential]['resistance'])
        if (len(export_array) == 1):
            if (export_array[0] != zero_potential):
                conductivity_matrix[potential][export_array[0]] -= 1 / (graph[potential][export_array[0]]['resistance'])
                current_matrix[potential] -= (graph[potential][export_array[0]]['voltage']) / (
                graph[potential][export_array[0]]['resistance'])
        else:
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
        else:
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

    for branch in graph.edges():
        print(graph.edges[branch])

# end functions and support elements

# running algorithm

# teseted
# teseted
# teseted
# teseted

"""
#directed graph

matrix = func_list_to_matrix(directed_adjacency_list)
nodes = func_count_of_nodes(matrix)
branches = func_count_of_branches(directed_adjacency_list)
#graph = func_initialization(directed_adjacency_list, nodes, branches)
graph = func_initialization_directed_graph(directed_adjacency_list, nodes)
func_spanning_trees(graph)
"""

"""
#undirected graph

list = func_directed_to_nondirected_adjacency_list(directed_adjacency_list)
matrix = func_list_to_matrix(list)
nodes = func_count_of_nodes(matrix)
branches = func_count_of_branches(list)
graph = func_initialization_undirected_graph(list, nodes)

func_spanning_trees(graph)
"""


# teseted
# teseted
# teseted
# teseted

