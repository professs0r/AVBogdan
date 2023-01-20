
import random
import test_node_potential_algo as algo
import numpy as np
import networkx as nx
import copy

"""
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
"""

# маленькая кошка
"""
COUNT_NODES = 10

# edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated), PS='E1' (source energy),
#          type_edge='Source or Branch or Load or Chord')

edge_0 = (0, 1, 0.1, 630, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Source')
edge_1 = (1, 2, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_2 = (1, 4, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_3 = (2, 3, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_4 = (2, 5, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_5 = (3, 6, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_6 = (4, 5, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_7 = (4, 7, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_8 = (5, 6, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_9 = (5, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_10 = (6, 9, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_11 = (7, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_12 = (8, 9, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')


edge_nagr_2 = (2, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_3 = (3, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_4 = (4, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_5 = (5, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_6 = (6, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_7 = (7, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_8 = (8, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_9 = (9, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')

edges_lines = np.array([edge_0,
                        edge_1,
                        edge_2,
                        edge_3,
                        edge_4,
                        edge_5,
                        edge_6,
                        edge_7,
                        edge_8,
                        edge_9,
                        edge_10,
                        edge_11,
                        edge_12])
edges_nagr = np.array([edge_nagr_2,
                       edge_nagr_3,
                       edge_nagr_4,
                       edge_nagr_5,
                       edge_nagr_6,
                       edge_nagr_7,
                       edge_nagr_8,
                       edge_nagr_9])

# для расчёта токов
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
                  edge_10,
                  edge_11,
                  edge_12,
                  edge_nagr_2,
                  edge_nagr_3,
                  edge_nagr_4,
                  edge_nagr_5,
                  edge_nagr_6,
                  edge_nagr_7,
                  edge_nagr_8,
                  edge_nagr_9])
"""
# маленькая кошка

# большая кошка

COUNT_NODES = 17

# edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated), PS='E1' (source energy))

edge_0 = (0, 1, 0.1, 1000, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Source')
edge_1 = (1, 2, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_2 = (1, 5, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_3 = (2, 3, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_4 = (2, 6, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_5 = (3, 4, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_6 = (3, 7, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_7 = (4, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_8 = (5, 6, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_9 = (5, 9, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10', 'Chord')
edge_10 = (6, 7, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_11 = (6, 10, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10', 'Chord')
edge_12 = (7, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch')
edge_13 = (7, 11, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10', 'Chord')
edge_14 = (8, 12, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10', 'Chord')
edge_15 = (9, 10, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_16 = (9, 13, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_17 = (10, 11, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_18 = (10, 14, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_19 = (11, 12, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_20 = (11, 15, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_21 = (12, 16, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_22 = (13, 14, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_23 = (14, 15, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')
edge_24 = (15, 16, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Branch')

edge_nagr_2 = (2, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_3 = (3, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_4 = (4, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_5 = (5, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_6 = (6, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_7 = (7, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_8 = (8, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load')
edge_nagr_9 = (9, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_10 = (10, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_11 = (11, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_12 = (12, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_13 = (13, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_14 = (14, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_15 = (15, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')
edge_nagr_16 = (16, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2', 'Load')

#                                        START SOURCE DATA
#                                        START SOURCE DATA
#                                        START SOURCE DATA

#                                        START WORKING ALGO
#                                        START WORKING ALGO
#                                        START WORKING ALGO
edges_lines = np.array([edge_0,
                        edge_1,
                        edge_2,
                        edge_3,
                        edge_4,
                        edge_5,
                        edge_6,
                        edge_7,
                        edge_8,
                        edge_9,
                        edge_10,
                        edge_11,
                        edge_12,
                        edge_13,
                        edge_14,
                        edge_15,
                        edge_16,
                        edge_17,
                        edge_18,
                        edge_19,
                        edge_20,
                        edge_21,
                        edge_22,
                        edge_23,
                        edge_24])
edges_nagr = np.array([edge_nagr_2,
                       edge_nagr_3,
                       edge_nagr_4,
                       edge_nagr_5,
                       edge_nagr_6,
                       edge_nagr_7,
                       edge_nagr_8,
                       edge_nagr_9,
                       edge_nagr_10,
                       edge_nagr_11,
                       edge_nagr_12,
                       edge_nagr_13,
                       edge_nagr_14,
                       edge_nagr_15,
                       edge_nagr_16])

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
                  edge_10,
                  edge_11,
                  edge_12,
                  edge_13,
                  edge_14,
                  edge_15,
                  edge_16,
                  edge_17,
                  edge_18,
                  edge_19,
                  edge_20,
                  edge_21,
                  edge_22,
                  edge_23,
                  edge_24,
                  edge_nagr_2,
                  edge_nagr_3,
                  edge_nagr_4,
                  edge_nagr_5,
                  edge_nagr_6,
                  edge_nagr_7,
                  edge_nagr_8,
                  edge_nagr_9,
                  edge_nagr_10,
                  edge_nagr_11,
                  edge_nagr_12,
                  edge_nagr_13,
                  edge_nagr_14,
                  edge_nagr_15,
                  edge_nagr_16])

# большая кошка

"""
algo.func_calculated_current_node_potential_algo(graph)
print("Величины токов в исходном графе: ")
for branch in graph.edges():
    print(branch)
    print(graph.edges[branch]['I'])
"""

"""
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
"""

"""
SUPPORT FUNCTIONS
"""

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
    while to_explore:
        next = to_explore.pop(0)
        new_vertexes = [i for i in graph[next] if (i not in visited and graph.edges[(next, i)]['type_edge'] != "Load")]
        # НЕОБХОДИМО ДОБАВИТЬ ПРОВЕРКУ НА ДОСРОЧНОЕ УСПЕШНОЕ ВЫПОЛНЕНИЕ ПРОВЕРКИ НА СВЯЗНОСТЬ
        # НЕОБХОДИМО ДОБАВИТЬ ПРОВЕРКУ НА ДОСРОЧНОЕ УСПЕШНОЕ ВЫПОЛНЕНИЕ ПРОВЕРКИ НА СВЯЗНОСТЬ
        # НЕОБХОДИМО ДОБАВИТЬ ПРОВЕРКУ НА ДОСРОЧНОЕ УСПЕШНОЕ ВЫПОЛНЕНИЕ ПРОВЕРКИ НА СВЯЗНОСТЬ
        for i in new_vertexes:
            lens[i] = lens[next] + 1
        to_explore.extend(new_vertexes)
        visited.update(new_vertexes)
    return lens

def is_connected(G):
    """
    стандартная функция для проверки графа на связность (использует в себе алгоритм обхода графа в ширину)
    :param G:
    :return:
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined ", "for the null graph."
        )
    result = sum(1 for node in func_BFS(G, 0)) == len(G)
    return result

def difference_of_lists(argument1, argument2):
    """
    функция для вычитания одного списка рёбер из другого, например: self.__local_best_position - self.__current_position
    :param argument1: эталон - из которого вычитают
    :param argument2: преверяемый - который вычитается
    :return:
    """
    result_list = []
    for edge in argument2:
        if edge not in argument1:
            result_list.append(edge)
    if not result_list:
        # список пуст возвращаем ничего (None)
        return None
    return result_list

"""
SUPPORT FUNCTIONS
"""

class Floatbee:
    """
    Класс пчёл, где в качестве координат используются числа с плавающей точкой
    """
    def __init__(self, graph):
        self.__local_graph = graph.copy()
        self.__primal_graph = graph.copy()
        self.__count_edges = self.__count_edges_for_remove()
        if self.__check_graph():
            self.__position = self.__get_initial_position()
            self.fitness = self.calc_fitness()
        else:
            print("Невозможно больше удалять рёбра из графа. Остовное дерево получено.")

    @staticmethod
    def get_start_range():
        """

        :return:
        """
        return 0.25  # радиус, в пределах которого будет проверяться наличие других пчёлок

    @staticmethod
    def get_range_koeff():
        """

        :return:
        """
        return 0.98

    def get_count_edges(self):
        """

        :return:
        """
        return self.__count_edges

    def __list_connectivity_graph(self, list_of_edges):
        """
        список рёбер, которые можно удалить из графа и связность не нарушится
        :param list_of_edges:
        :return:
        """
        for edge in list_of_edges:
            if not self.__check_connectivity_graph(edge[0]):
                list_of_edges.remove(edge)
        return list_of_edges

    def __check_connectivity_graph(self, edge_for_remove):
        """
        функция для проверки связности графа
        :param edge_for_remove:
        :return:
        """
        temp_edge = list(self.__local_graph.edges[edge_for_remove].values())
        self.__local_graph.remove_edge(edge_for_remove[0], edge_for_remove[1])
        Flag = False
        if is_connected(self.__local_graph):
            Flag = True
        self.__local_graph.add_edge(int(edge_for_remove[0]), int(edge_for_remove[1]), resistance=float(temp_edge[0]),
                                    voltage=float(temp_edge[1]), type=int(temp_edge[2]),
                                    length=float(temp_edge[3]),
                                    cross_section=float(temp_edge[4]), I=float(temp_edge[5]),
                                    material=temp_edge[6],
                                    r_0=float(temp_edge[7]), x_0=float(temp_edge[8]),
                                    cos_y=float(temp_edge[9]),
                                    sin_y=float(temp_edge[10]), lose_volt=float(temp_edge[11]),
                                    lose_energy=float(temp_edge[12]), PS=str(temp_edge[13]),
                                    type_edge=str(temp_edge[14]))
        return Flag

    def __check_graph(self):
        """
        функция для проверки возможности удаления ребра из графа
        если количество ветвей равно числу узлов минус единица, то уже найдено остовное дерево и больше удалять
        рёбра нельзя
        :return:
        """
        counter_of_branches = 0
        for edge in self.__local_graph.edges():
            if str(self.__local_graph.edges[edge]['type_edge']) == str('Branch'):
                counter_of_branches += 1
        if counter_of_branches == int(self.__local_graph.number_of_nodes() - 1):
            return False
        return True

    def __count_edges_for_remove(self):
        """
        количество рёбер, которые нужно удалить из графа, чтобы получилось остовное дерево
        (для размерности position)
        :return:
        """
        count_of_edges = 0
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load'):
                count_of_edges += 1
        return int(count_of_edges - (int(self.__local_graph.number_of_nodes()) - 1))

    def __get_initial_position(self):
        """

        :param swarm:
        :return:
        """
        count_edges = copy.deepcopy(self.__count_edges)
        position = []
        while count_edges:
            algo.func_calculated_current_node_potential_algo(self.__local_graph)
            list_edges = []
            for edge in self.__local_graph.edges():
                if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                    'type_edge'] != str('Source'):
                    if self.__check_connectivity_graph(edge):
                        list_edges.append([edge, self.__local_graph.edges[edge]['I']])
            sort_array = algo.quick_sort(list_edges)
            number_of_excess_edges = int(len(sort_array) / 4)
            for num in range(number_of_excess_edges):
                sort_array.pop()
            del number_of_excess_edges
            correct_list_edges = self.__list_connectivity_graph(sort_array)
            sum_of_currents = 0.0
            max_limit_for_position = 0
            for iter in range(2):
                index = 0
                for edge in correct_list_edges:
                    if iter:
                        max_limit_for_position += (1 - abs(edge[1] / sum_of_currents))
                        correct_list_edges[index].append(1 - abs(edge[1] / sum_of_currents))
                        index += 1
                    else:
                        sum_of_currents += abs(edge[1])
            sup_variable = correct_list_edges[0][2]
            roulette = np.random.uniform(0.0, float(max_limit_for_position))
            for item in correct_list_edges:
                if roulette > sup_variable:
                    sup_variable += item[2]
                else:
                    position.append(item[0])
                    self.__local_graph.remove_edge(item[0][0], item[0][1])
                    count_edges -= 1
                    break
        return position

    def pop_edge_by_index_insert_new_edge(self, index):
        """
        функция для удаление из self.__position ребра по индексу index и добавления нового ребра
        иначе: в граф обратно добавляет ребро соответствующее индексу index и удаляет совершенного другое
        с проверкой на связность
        :param index:
        :return:
        """
        adding_edge = list(self.__position[index])
        self.__position.pop(index)
        for edge in edges_lines:
            if int(adding_edge[0]) == int(edge[0]) and int(adding_edge[1]) == int(edge[1]):
                self.__local_graph.add_edge(int(edge[0]), int(edge[1]),
                                            resistance=float(edge[2]),
                                            voltage=float(edge[3]), type=int(edge[4]),
                                            length=float(edge[5]),
                                            cross_section=float(edge[6]), I=float(edge[7]),
                                            material=edge[8],
                                            r_0=float(edge[9]), x_0=float(edge[10]),
                                            cos_y=float(edge[11]),
                                            sin_y=float(edge[12]), lose_volt=float(edge[13]),
                                            lose_energy=float(edge[14]), PS=str(edge[15]),
                                            type_edge=str(edge[16]))
        list_of_edges_for_check = []
        for edge in self.__local_graph.edges():
            if str(self.__local_graph.edges[edge]['type_edge']) == str('Branch'):
                if self.__check_connectivity_graph(edge) and list(edge) != list(adding_edge):
                    list_of_edges_for_check.append(edge)
        index_random_removing_edge = random.randint(0, len(list_of_edges_for_check) - 1)
        self.__position.append(list_of_edges_for_check[index_random_removing_edge])
        self.__local_graph.remove_edge(list_of_edges_for_check[index_random_removing_edge][0], list_of_edges_for_check[index_random_removing_edge][1])
        algo.func_calculated_current_node_potential_algo(self.__local_graph)


    def calc_fitness(self):
        """
        Расчёт целевой функции. Этот метод необходимо перегрузить в производном классе.
        Функция не возвращает значение целевой функции, а только устанавливает член self.__fitness.
        Эту функцию необходимо вызывать после каждого изменения координаты пчелы.
        :return:
        """
        algo.func_calculated_current_node_potential_algo(self.__local_graph)
        fitness = algo.func_law_Joule_Lenz(self.__local_graph)
        return fitness

    def sort(self, otherbee):
        """
        Функция для сортировки пчёл по их целевой функции (здоровью) в порядке убывания
        :param otherbee:
        :return:
        """
        if self.fitness < otherbee.fitness:
            return -1
        elif self.fitness > otherbee.fitness:
            return 1
        else:
            return 0

    def otherpatch(self, bee_list, range_list):
        """
        Проверить находится ли пчела на том же участке, что и одна из пчёл bee_list.
        :param beelist: список лучших мест
        :param rangelist: интервал изменения каждой из координат
        :return:
        """
        if len(bee_list) == 0:
            return True
        for curr_bee in bee_list:
            position = curr_bee.get_position()
            if difference_of_lists(self.__position, position) is None or len(difference_of_lists(self.__position, position)) <= int(len(position)*range_list):
                return False
        return True

    def get_position(self):
        """
        Вернуть копию!!!!! своих координат.
        :return:
        """
        return self.__position

    def goto(self, other_pos):
        """
        В исходниках прототип функции выглядел следующим образом: def goto(self, other_pos, range_list)
        Перелёт в окрестности места, которое нашла другая пчела. Не в то же самое место!
        :param other_pos:
        :param range_list:
        :return:
        """
        # К каждой из координат добавляет случайное значение
        if difference_of_lists(other_pos, self.__position) is None:
            diff = 0
        else:
            # ранее уже вычислили значение функции (надо использовать через переменную
            diff = random.randint(0, len(difference_of_lists(other_pos, self.__position))-1)
        self.pop_edge_by_index_insert_new_edge(diff)
        # Рассчитаем и сохраним целевую функцию.
        self.fitness = self.calc_fitness()

    def goto_random(self):
        """

        :return:
        """
        # Заполним координаты случайными значениями
        self.__local_graph = self.__primal_graph.copy()
        self.__position = self.__get_initial_position()
        self.fitness = self.calc_fitness()

class Hive:
    """
    Класс описывающий улей. Улей управляет пчёлами.
    """
    def __init__(self, scout_bee_count, selected_bee_count, best_bee_count, sel_sites_count, best_sites_count,
                 range_list, bee_type, graph):
        """

        :param scout_bee_count: количество пчёл-разведчиков
        :param best_bee_count: количество пчёл, посылаемых на один из лучших участков
        :param selected_bee_count: количество пчёл, посылаемых на остальные выбранные участки
        :param sel_sites_count: количество выбранных участков
        :param bet_sites_count: количество лучших участков среди выбранных
        :param range_list: список диапазонов координат для одного участка
        :param bee_type: класс пчелы, производный от bee
        """
        self.scout_bee_count = scout_bee_count
        self.selected_bee_count = selected_bee_count
        self.best_bee_count = best_bee_count
        self.sel_sites_count = sel_sites_count
        self.best_sites_count = best_sites_count
        self.range_list = range_list
        self.bee_type = bee_type
        self.__local_graph = graph.copy()

        # Лучшая на данный момент позиция
        self.best_position = None
        # Лучшее на данный момент здоровье пчелы (чем больше, тем лучше)
        self.best_fitness = -1.0e9
        # Начальное заполнение роя пчёлами со случайными координатами
        bee_count = scout_bee_count + selected_bee_count * sel_sites_count + best_bee_count * best_sites_count
        # Строка кода ниже изменена мною
        self.swarm = [Floatbee(self.__local_graph) for n in range(bee_count)]

        # Лучшие и выбранные места
        self.best_sites = []
        self.sel_sites = []

        self.swarm = sorted(self.swarm, key=lambda func: func.fitness, reverse=False)
        self.best_position = self.swarm[0].get_position()
        self.best_fitness = self.swarm[0].fitness

    def send_bees(self, position, index, count):
        """
        Послать пчёл на позицию. Функция возвращает номер следующей пчелы для вылета. ТРЕБУЕТСЯ ДОРАБОТКА ЭТОЙ ФУНКЦИИ, ЧТОБЫ ОТПРАВЛЯЛОСЬ НЕОБХОДИМОЕ КОЛИЧЕСТВО ПЧЁЛОК НА КАЖДЫЕ ИЗ ВЫБРАННЫХ И ЛУЧШИХ УЧАСТКОВ
        :return:
        """
        for n in range(count):
            # Чтобы не выйти за пределы улея.
            if index == len(self.swarm):
                break
            curr_bee = self.swarm[index]
            if curr_bee not in self.best_sites and curr_bee not in self.sel_sites:
                # Пчела не на лучших или выбранных позициях
                curr_bee.goto(position)
            index += 1
        return index

    def next_step(self):
        """
        Функция, в которой описывается логика новой итерации.
        :return:
        """
        # Выбираем самые лучшие места и сохраняем ссылки на тех, кто их нашёл
        self.best_sites = [self.swarm[0]]
        curr_index = 1
        for curr_bee in self.swarm[curr_index:-1]:
            # Если пчела находится в пределах уже отмеченного лучшего участка, то её положение на считаем.
            if curr_bee.otherpatch(self.best_sites, self.range_list):
                self.best_sites.append(curr_bee)
                if len(self.best_sites) == self.best_sites_count:
                    # строку: "curr_index += 1" я добавил (в исходниках её не было)
                    curr_index += 1
                    break
            curr_index += 1
        self.sel_sites = []
        for curr_bee in self.swarm[curr_index:-1]:
            if curr_bee.otherpatch(self.best_sites, self.range_list) and curr_bee.otherpatch(self.sel_sites, self.range_list):
                self.sel_sites.append(curr_bee)
                if len(self.sel_sites) == self.sel_sites_count:
                    break
        # Отправляем пчёл на задание
        # Отправляем сначала на лучшие места
        # Номер очередной отправляемой пчелы. 0-ую пчелу никуда не отправляем.
        bee_index = 1
        for best_bee in self.best_sites:
            bee_index = self.send_bees(best_bee.get_position(), bee_index, self.best_bee_count)
        for sel_bee in self.sel_sites:
            bee_index = self.send_bees(sel_bee.get_position(), bee_index, self.selected_bee_count)
        # Оставшихся пчёл отправим куда попало
        # строка кода ниже - та, что была в исходниках, а через строку изменённая мною строка (БЕСТОЛОЧЬ!! НАДО НЕ -2 ПИСАТЬ!!!!)
        #for curr_bee in self.swarm[bee_index:-1]:
        for curr_bee in self.swarm[bee_index:-2]:
            curr_bee.goto_random()
        self.swarm = sorted(self.swarm, key=lambda func: func.fitness, reverse=True)
        self.best_position = self.swarm[-1].get_position()
        self.best_fitness = self.swarm[-1].fitness




def run_ABC():
    """
    Функция для запуска алгоритма.
    :return:
    """

    #################################################################################
    #################################################################################
    #################################################################################
    #########################   ПАРАМЕТРЫ АЛГОРИТМА    ##############################
    #################################################################################
    #################################################################################
    #################################################################################

    graph = algo.func_edges_to_undirected_graph(edges, COUNT_NODES)

    # Класс пчёл, который будет использоваться в алгоритме.
    bee_type = Floatbee(graph)
    # Количество пчёл-разведчиков
    #scout_bee_count = 300
    scout_bee_count = 30
    #scout_bee_count = 2
    # Количество пчёл, отправляемых на выбранные, но не лучшие участки
    # не влияет на работоспособность программы
    #selected_bee_count = 10
    selected_bee_count = 5
    #selected_bee_count = 2
    # Количество пчёл, отправляемых на лучшие участки
    # не влияет на работоспособность программы
    #best_bee_count = 30
    best_bee_count = 10
    #best_bee_count = 2
    # Количество выбранных, но не лучших участков
    # ВЛИЯЕТ на работоспособность программы
    # или НЕ влияет на работоспособность программы
    #sel_sites_count = 15
    sel_sites_count = 5
    #sel_sites_count = 2
    # Количество лучших участков
    # ВЛИЯЕТ на работоспособность программы
    # или НЕ влияет на работоспособность программы
    best_sites_count = 5
    #best_sites_count = 2
    # Количество запусков алгоритма
    # ВЛИЯЕТ на работоспособность программы
    # слабо влияет на сходимость
    # или НЕ влияет на работоспособность программы
    #run_count = 10
    #run_count = 3
    run_count = 1
    # Максимальное количество итераций
    # не влияет на работоспособность программы
    # ОЧЕНЬ СИЛЬНО ВЛИЯЕТ на работоспособность программы
    #max_iteration = 500
    #max_iteration = 50
    max_iteration = 10
    # Через такое количество итераций без нахождения лучшего решения уменьшим область поиска
    # ВЛИЯЕТ на работоспособность программы
    # или НЕ влияет на работоспособность программы
    #max_func_counter = 3
    #max_func_counter = 2
    max_func_counter = 1
    # Во столько раз будет уменьшаться область поиска
    koeff = bee_type.get_range_koeff()

    #################################################################################
    #################################################################################
    #################################################################################
    #########################   ПАРАМЕТРЫ АЛГОРИТМА    ##############################
    #################################################################################
    #################################################################################
    #################################################################################

    for run_number in range(run_count):
        curr_hive = Hive(scout_bee_count, selected_bee_count, best_bee_count, sel_sites_count, best_sites_count,
                        bee_type.get_start_range(), bee_type, graph)
        # Начальное значение целевой функции
        best_func = -1.0e9
        # Количество итераций без улучшения целевой функции
        func_counter = 0
        for n in range(max_iteration):
            curr_hive.next_step()
            if curr_hive.best_fitness != best_func:
                # Найдено место, где целевая функция лучше.
                best_func = curr_hive.best_fitness
                func_counter = 0
                # Обновновляем рисунок роя пчёл
                #plot_swarm(curr_hive, 0, 1)
                print("\n*** Iteration %d  / %d" % (run_number+1, n))
                print("Best Position: %s" % curr_hive.best_position)
                print("Best fitness: %f" % curr_hive.best_fitness)
            else:
                func_counter += 1
                if func_counter == max_func_counter:
                    # Уменьшаем размер участков
                    curr_hive.range_list *= koeff
                    func_counter = 0
                    print("\n*** Iteration %d / %d (new range" % (run_number+1, n))
                    print("New range: %s" % curr_hive.range_list)
                    print("Best Position: %s" % curr_hive.best_position)
                    print("Best Fitness: %f" % curr_hive.best_fitness)

run_ABC()

print("\nЕсли ты это читаешь, значит, программа отработала!")