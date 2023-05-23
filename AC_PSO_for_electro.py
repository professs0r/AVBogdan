import networkx as nx
import numpy as np
from abc import abstractmethod
import math
import cmath
import test_node_potential_algo as algo


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
    :param argument1:
    :param argument2:
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
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
"""
"""
# Переменная кошка с 17 котятами

COUNT_NODES = 17

# edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated))

edge_0 = (0, 1, complex(0.1, 0.047), complex(630, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Source', complex(0, 0))
edge_1 = (1, 2, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_2 = (1, 5, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_3 = (2, 3, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_4 = (2, 6, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_5 = (3, 4, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_6 = (3, 7, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_7 = (4, 8, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_8 = (5, 6, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_9 = (5, 9, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_10 = (6, 7, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_11 = (6, 10, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_12 = (7, 8, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_13 = (7, 11, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_14 = (8, 12, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_15 = (9, 10, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_16 = (9, 13, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_17 = (10, 11, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_18 = (10, 14, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_19 = (11, 12, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_20 = (11, 15, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_21 = (12, 16, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_22 = (13, 14, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_23 = (14, 15, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_24 = (15, 16, complex(0.5, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))

# в нагрузочных ветвях: в сопротивлениях у мнимой части убрал минус
edge_nagr_1 = (2, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_2 = (3, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_3 = (4, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_4 = (5, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_5 = (6, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_6 = (7, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_7 = (8, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_8 = (9, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_9 = (10, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_10 = (11, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_11 = (12, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
#edge_nagr_11 = (12, 0, complex(23.178, 21.59), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_12 = (13, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_13 = (14, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_14 = (15, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
#edge_nagr_14 = (15, 0, complex(128.008, -26.9892), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))
edge_nagr_15 = (16, 0, complex(127.008, 95.256), 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(2000, 1500))

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
edges_nagr = np.array([edge_nagr_1,
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
                       edge_nagr_15])

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
                  edge_nagr_1,
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
                  edge_nagr_15])
"""

# кошка с 33 котятами
COUNT_NODES = 33
# edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated), PS='E1' (source energy),
#          type_edge='Source or Branch or Load or Chord',
#          power=140e3 [for load in W, for source in WA])

edge_line_0_1 = (0, 1, complex(0.0922, 0.047), complex(12.66e3, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Source', complex(100e6,0))
edge_line_1_2 = (1, 2, complex(0.493, 0.2511), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_2_3 = (2, 3, complex(0.366, 0.1864), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_3_4 = (3, 4, complex(0.3811, 0.1941), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_4_5 = (4, 5, complex(0.819, 0.707), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_5_6 = (5, 6, complex(0.01872, 0.6188), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_6_7 = (6, 7, complex(0.7114, 0.2351), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_7_8 = (7, 8, complex(1.03, 0.74), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_8_9 = (8, 9, complex(1.044, 0.74), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_9_10 = (9, 10, complex(0.1966, 0.065), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_10_11 = (10, 11, complex(0.3744, 0.1238), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_11_12 = (11, 12, complex(1.468, 1.155), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_12_13 = (12, 13, complex(0.5416, 0.7129), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_13_14 = (13, 14, complex(0.591, 0.526), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_14_15 = (14, 15, complex(0.7463, 0.0545), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_15_16 = (15, 16, complex(1.289, 1.721), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_16_17 = (16, 17, complex(0.732, 0.574), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_1_18 = (1, 18, complex(0.164, 0.1565), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_18_19 = (18, 19, complex(1.5042, 1.3554), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_19_20 = (19, 20, complex(0.4095, 0.4784), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_20_21 = (20, 21, complex(0.7089, 0.9373), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_2_22 = (2, 22, complex(0.04512, 0.3083), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_22_23 = (22, 23, complex(0.898, 0.7091), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_23_24 = (23, 24, complex(0.896, 0.7011), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_5_25 = (5, 25, complex(0.203, 0.1034), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_25_26 = (25, 26, complex(0.2042, 0.1447), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_26_27 = (26, 27, complex(1.059, 0.9337), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_27_28 = (27, 28, complex(0.8042, 0.7006), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_28_29 = (28, 29, complex(0.5075, 0.2585), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_29_30 = (29, 30, complex(0.9744, 0.963), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_30_31 = (30, 31, complex(0.3105, 0.3619), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_31_32 = (31, 32, complex(0.341, 0.5302), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_7_20 = (7, 20, complex(2.0, 2.0), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_8_14 = (8, 14, complex(2.0, 2.0), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_11_21 = (11, 21, complex(2.0, 2.0), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_17_32 = (17, 32, complex(0.5, 0.5), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))
edge_line_24_28 = (24, 28, complex(0.5, 0.5), complex(0, 0), 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', complex(0, 0))

edge_nagr_2 = (2, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 40e3))
edge_nagr_3 = (3, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(120e3, 80e3))
edge_nagr_4 = (4, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 30e3))
edge_nagr_5 = (5, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 20e3))
edge_nagr_6 = (6, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(200e3, 100e3))
edge_nagr_7 = (7, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(200e3, 100e3))
edge_nagr_8 = (8, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 20e3))
edge_nagr_9 = (9, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 20e3))
edge_nagr_10 = (10, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(45e3, 30e3))
edge_nagr_11 = (11, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 35e3))
edge_nagr_12 = (12, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 35e3))
edge_nagr_13 = (13, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(120e3, 80e3))
edge_nagr_14 = (14, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 10e3))
edge_nagr_15 = (15, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 20e3))
edge_nagr_16 = (16, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 20e3))
edge_nagr_17 = (17, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 40e3))
edge_nagr_18 = (18, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 40e3))
edge_nagr_19 = (19, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 40e3))
edge_nagr_20 = (20, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 40e3))
edge_nagr_21 = (21, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 40e3))
edge_nagr_22 = (22, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(90e3, 50e3))
edge_nagr_23 = (23, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(420e3, 200e3))
edge_nagr_24 = (24, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(420e3, 200e3))
edge_nagr_25 = (25, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 25e3))
edge_nagr_26 = (26, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 25e3))
edge_nagr_27 = (27, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 20e3))
edge_nagr_28 = (28, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(120e3, 70e3))
edge_nagr_29 = (29, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(200e3, 600e3))
edge_nagr_30 = (30, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(150e3, 70e3))
edge_nagr_31 = (31, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(210e3, 100e3))
edge_nagr_32 = (32, 0, complex(0, 0), complex(0, 0), 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', complex(60e3, 40e3))


edges_lines = np.array([edge_line_0_1,
                        edge_line_1_2,
                        edge_line_2_3,
                        edge_line_3_4,
                        edge_line_4_5,
                        edge_line_5_6,
                        edge_line_6_7,
                        edge_line_7_8,
                        edge_line_8_9,
                        edge_line_9_10,
                        edge_line_10_11,
                        edge_line_11_12,
                        edge_line_12_13,
                        edge_line_13_14,
                        edge_line_14_15,
                        edge_line_15_16,
                        edge_line_16_17,
                        edge_line_1_18,
                        edge_line_18_19,
                        edge_line_19_20,
                        edge_line_20_21,
                        edge_line_2_22,
                        edge_line_22_23,
                        edge_line_23_24,
                        edge_line_5_25,
                        edge_line_25_26,
                        edge_line_26_27,
                        edge_line_27_28,
                        edge_line_28_29,
                        edge_line_29_30,
                        edge_line_30_31,
                        edge_line_31_32,
                        edge_line_7_20,
                        edge_line_8_14,
                        edge_line_11_21,
                        edge_line_17_32,
                        edge_line_24_28
                        ])
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
                       edge_nagr_16,
                       edge_nagr_17,
                       edge_nagr_18,
                       edge_nagr_19,
                       edge_nagr_20,
                       edge_nagr_21,
                       edge_nagr_22,
                       edge_nagr_23,
                       edge_nagr_24,
                       edge_nagr_25,
                       edge_nagr_26,
                       edge_nagr_27,
                       edge_nagr_28,
                       edge_nagr_29,
                       edge_nagr_30,
                       edge_nagr_31,
                       edge_nagr_32])

# для расчёта токов
edges = np.array([edge_line_0_1,
                  edge_line_1_2,
                  edge_line_2_3,
                  edge_line_3_4,
                  edge_line_4_5,
                  edge_line_5_6,
                  edge_line_6_7,
                  edge_line_7_8,
                  edge_line_8_9,
                  edge_line_9_10,
                  edge_line_10_11,
                  edge_line_11_12,
                  edge_line_12_13,
                  edge_line_13_14,
                  edge_line_14_15,
                  edge_line_15_16,
                  edge_line_16_17,
                  edge_line_1_18,
                  edge_line_18_19,
                  edge_line_19_20,
                  edge_line_20_21,
                  edge_line_2_22,
                  edge_line_22_23,
                  edge_line_23_24,
                  edge_line_5_25,
                  edge_line_25_26,
                  edge_line_26_27,
                  edge_line_27_28,
                  edge_line_28_29,
                  edge_line_29_30,
                  edge_line_30_31,
                  edge_line_31_32,
                  edge_line_7_20,
                  edge_line_8_14,
                  edge_line_11_21,
                  edge_line_17_32,
                  edge_line_24_28,
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
                  edge_nagr_16,
                  edge_nagr_17,
                  edge_nagr_18,
                  edge_nagr_19,
                  edge_nagr_20,
                  edge_nagr_21,
                  edge_nagr_22,
                  edge_nagr_23,
                  edge_nagr_24,
                  edge_nagr_25,
                  edge_nagr_26,
                  edge_nagr_27,
                  edge_nagr_28,
                  edge_nagr_29,
                  edge_nagr_30,
                  edge_nagr_31,
                  edge_nagr_32])


"""
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
"""


"""
SUPPORT FUNCTIONS
"""

def func_find_edge_in_graph(graph, list_edges):
    """
    функция, которая из списка рёбер находит каждое ребро в самом графе (велосипедный вариант словаря)
    :param graph:
    :param list_edges:
    :return:
    """
    for item in list_edges:
        if tuple(item[0]) in graph.edges():
            print("item[1] = ", item[1])

def func_set_grade_limmit():
    """
    :return:
    FIX ME!!
    """
    print("Необходимо задать количество параметров, в пределах которых производиться оптимизация.")
    count_limit = input("Введите число параметров: ")
    try:
        count_limit = int(count_limit)
    except ValueError:
        print("Введите число!")

def print_result(swarm, iteration):
    template = u"""Iteration: {iter}
    Best Position: {best_position}
    Best Final Func: {best_final_func}
    """
    result = template.format(iter=iteration,
                             best_position = swarm.global_Best_Position,
                             best_final_func = swarm.global_Best_Position)
    return result

"""
SUPPORT FUNCTIONS
"""
class Swarm:
    """
    Класс, описывающий весь рой вцелом
    """
    def __init__(self, graph, swarm_size, constant_K, constant_C1, constant_C2):
        assert (constant_C1 + constant_C2) > 4, "Сумма коэффициентов C1 и C2 должна быть больше 4!"
        self.__constant_C1 = constant_C1
        self.__constant_C2 = constant_C2
        self.__constant_K = constant_K
        self.__constant_PHI = constant_C1 + constant_C2
        self.__constant_CHI = self.__get_constant_CHI(self.__constant_PHI, constant_K)
        #print("self.__constant_CHI = ", self.__constant_CHI)
        self.__swarm_size = swarm_size
        self.__global_Best_Final_func = None
        self.__global_Best_position = None
        self.__grade_of_change = self.__grade_of_change(graph)
        self.__swarm = self.__create_swarm()

    def __get_constant_CHI(self, constnt_PHI, constant_K):
        return float(2*constant_K / (abs(2 - constnt_PHI - math.sqrt(constnt_PHI**2 - 4*constnt_PHI))))

    def __grade_of_change(self, graph):
        """

        :return:
        """
        counter_of_branches = 0
        for edge in graph.edges():
            if str(graph.edges[edge]['type_edge']) != str('Load'):
                counter_of_branches += 1
        return counter_of_branches - (graph.number_of_nodes() - 1)

    def __create_swarm(self):
        """
        Инициализируем рой
        :return:
        """
        return [Particle(self) for _ in range(self.__swarm_size)]

    def get_final_func(self, position, local_graph):
        """
        в качестве аргумента функции вместо "position" попробую использовать сам граф
        :param position:
        :return:
        """
        #print("Рёбра, которые надо удалить: ", position)
        final_func = self._final_func(local_graph)
        if (self.global_Best_Final_func == None or cmath.polar(final_func).__getitem__(0) < cmath.polar(self.global_Best_Final_func).__getitem__(0)):
            self.__global_Best_Final_func = final_func
            #print("self.__global_Best_Final_func = ", self.__global_Best_Final_func)
            self.__global_Best_position = position.copy()
            #print("self.__global_Best_position = ", self.__global_Best_position)
        return final_func

    def next_iteration(self):
        """
        :return:
        """
        for particle in self.__swarm:
            particle.update_velocity_and_position(self)

    @abstractmethod
    def _final_func(self):
        pass

    @property
    def swarm_size(self):
        return self.__swarm_size

    @property
    def global_Best_Final_func(self):
        return self.__global_Best_Final_func

    @property
    def global_Best_Position(self):
        return self.__global_Best_position

    @property
    def constant_CHI(self):
        return self.__constant_CHI

    @property
    def constant_C1(self):
        return self.__constant_C1

    @property
    def constant_C2(self):
        return self.__constant_C2

    @property
    def constant_K(self):
        return self.__constant_K

    @property
    def grade_of_change(self):
        return self.__grade_of_change

class Particle(Swarm):
    """
    Класс, описывающий одну частицу
    """
    def __init__(self, swarm):
        self.__local_graph = graph.copy()
        self.__count_edges= self.__count_edges_for_remove()
        if self.__check_graph():
            self.__current_position = self.__get_initial_position(swarm)
            self.__velocity = self.__get_initial_velocity(swarm)
            #print("self.__velocity = ", self.__velocity)
            self.__local_Best_position = self.__current_position
            # в качестве параметра функции ниже попробую использовать сам граф, а не "position"
            self.__local_Best_Final_func = swarm.get_final_func(self.__current_position, self.__local_graph)
        else:
            print("Невозможно больше удалять рёбра из графа. Остовное дерево получено.")
            print("Для рассматриваемой частицы получено следующее значение потерь мощности: ", self.__local_Best_Final_func)
            print("Рёбра, которые необходимо оставить: ")
            for edge in self.__local_graph.edges():
                if str(self.__local_graph.edges[edge]['type_edge']) == str('Branch'):
                    print(edge)

    @property
    def position(self):
        return self.__current_position

    @property
    def velocity(self):
        return self.__velocity

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
        self.__local_graph.add_edge(int(edge_for_remove[0]), int(edge_for_remove[1]), resistance=temp_edge[0],
                                    voltage=temp_edge[1], type=int(temp_edge[2]),
                                    length=float(temp_edge[3]),
                                    cross_section=float(temp_edge[4]), I=temp_edge[5],
                                    material=temp_edge[6],
                                    r_0=float(temp_edge[7]), x_0=float(temp_edge[8]),
                                    cos_y=float(temp_edge[9]),
                                    sin_y=float(temp_edge[10]), lose_volt=temp_edge[11],
                                    lose_energy=temp_edge[12], PS=str(temp_edge[13]),
                                    type_edge=str(temp_edge[14]), power=temp_edge[15])
        return Flag

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

    def __get_initial_velocity(self, swarm):
        """
        функция возвращает массив значений размерностью равной "grade_limit" в пределах от "min_limit" до "max_limit"
        :param swarm:
        :return:
        """
        return np.random.randint(swarm.grade_of_change)

    def __get_initial_position(self, swarm):
        """
        везде меняю graph на self.__local_graph
        :param swarm:
        :return:
        """
        position = []
        while self.__count_edges:
            #algo.func_calculated_current_node_potential_algo_AC(self.__local_graph)
            algo.func_calculated_current_node_potential_algo_AC(self.__local_graph, True) # для заданных мощностей
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
                        max_limit_for_position += (1 - (cmath.polar(edge[1]).__getitem__(0) / sum_of_currents))
                        correct_list_edges[index].append(1 - (cmath.polar(edge[1]).__getitem__(0) / sum_of_currents))
                        index += 1
                    else:
                        sum_of_currents += cmath.polar(edge[1]).__getitem__(0)
            sup_variable = correct_list_edges[0][2]
            roulette = np.random.uniform(0.0, float(max_limit_for_position))
            for item in correct_list_edges:
                if roulette > sup_variable:
                    sup_variable += item[2]
                else:
                    position.append(item[0])
                    self.__local_graph.remove_edge(item[0][0], item[0][1])
                    self.__count_edges -= 1
                    break
        #print("position = ", position)
        #algo.func_calculated_current_node_potential_algo_AC(self.__local_graph)
        algo.func_calculated_current_node_potential_algo_AC(self.__local_graph, True)   # для заданных мощностей
        """
        print("Токи в ветвях после удаления хорд перечисленных выше.")
        for branch in self.__local_graph.edges():
            print(branch, " - ", cmath.polar(self.__local_graph.edges[branch]['I']).__getitem__(0))
        """
        """
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                'type_edge'] != str('Source'):
                print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        """
        return position

    def update_velocity_and_position(self, swarm):
        """
        :param swarm:
        :return:
        """
        #print("self.__local_Best_position = ", self.__local_Best_position)
        #print("self.__current_position = ", self.__current_position)
        # случайный вектор для коррекции скорости с учётом лучшей позиции данной частицы (rand() - во втором
        # слагаемом уравнения, сразу после константы C1)
        rand_current_Best_Position = np.random.random()
        # случайный вектор для коррекции скорости с учётом лучшей глобальной позиции всех частиц (Rand() - в третьем
        # слагаемом уравнения, сразу после константы C2)
        rand_global_Best_Position = np.random.random()
        # делим выражение для расчёта обновленого значения скорости на отдельные слагаемые
        # первое слагаемое с текущей сокростью частицы
        new_velocity_part_1 = [swarm.constant_CHI, self.__current_position[self.__velocity]]
        #print("new_velocity_part_1 = ", new_velocity_part_1)
        new_velocity_part_2 = [swarm.constant_CHI * swarm.constant_C1 * rand_current_Best_Position,
                                                    difference_of_lists(self.__local_Best_position, self.__current_position)]
        #print("new_velocity_part_2 = ", new_velocity_part_2)
        new_velocity_part_3 = [swarm.constant_CHI * swarm.constant_C2 * rand_global_Best_Position,
                                                    difference_of_lists(swarm.global_Best_Position, self.__current_position)]
        #print("new_velocity_part_3 = ", new_velocity_part_3)
        temp_velocity = []
        temp_velocity.append(new_velocity_part_1)
        temp_velocity.append(new_velocity_part_2)
        temp_velocity.append(new_velocity_part_3)
        #temp_velocity = list(new_velocity_part_1) + list(new_velocity_part_2) + list(new_velocity_part_3)
        #print("temp_velocity = ", temp_velocity)
        temp_edges, temp_values = self.edges_and_values(temp_velocity)
        #print("temp_edges = ", temp_edges)
        #print("temp_values = ", temp_values)
        if len(temp_edges) == 1:
            #print("Не интересует! Однозначно заменяется только одно ребро: ", temp_edges[0])
            #print("До удаления:")
            #print("self.__current_position = ", self.__current_position)
            self.__current_position.remove(temp_edges[0])
            #print("После удаления: ")
            #print("self.__current_position = ", self.__current_position)
            new_edge_for_position = self.add_old_edge_calculate_remove_new_edge(list(temp_edges[0]))
        else:
            #print("sum = ", sum(temp_values))
            roulette = np.random.uniform(0.0, float(sum(temp_values)))
            #print("roulette = ", roulette)
            sup_variable = 0
            index = 0
            for item in temp_values:
                sup_variable += item
                #print("sup_variable = ", sup_variable)
                if roulette < sup_variable:
                    # убираем выбранное ребро из position, добавляем выбранное ребро обратно в граф, производим расчёт
                    # параметров цепи, удаляем другое ребро с учётом того, что последнее удалённое неприкасаемо
                    #print("До удаления:")
                    #print("self.__current_position = ", self.__current_position)
                    #print("Удаляем ", temp_edges[index], ": ")
                    self.__current_position.remove(temp_edges[index])
                    #print("После удаления: ")
                    #print("self.__current_position = ", self.__current_position)
                    #self.add_old_edge_calculate_remove_new_edge(list(temp_edges[index]))
                    new_edge_for_position = self.add_old_edge_calculate_remove_new_edge(list(temp_edges[index]))
                    #print("Ребро, которое можно удалить = ", new_edge_for_position)
                    break
                index += 1
        self.__current_position.append(new_edge_for_position)
        self.__local_graph.remove_edge(new_edge_for_position[0], new_edge_for_position[1])
        #print("self.__current_position = ", self.__current_position)
        final_func = swarm.get_final_func(self.__current_position, self.__local_graph)
        #print("self.__local_Best_position", self.__local_Best_position)
        #print("self.__current_position", self.__current_position)
        #print("self.__local_Best_Final_func", self.__local_Best_Final_func)
        #print("final_func", final_func)
        if cmath.polar(final_func).__getitem__(0) < cmath.polar(self.__local_Best_Final_func).__getitem__(0):
            self.__local_Best_position = self.__current_position
            self.__local_Best_Final_func = final_func

    def edges_and_values(self, temp_velocity):
        """

        :return:
        """
        temp_edges = []
        for part in temp_velocity:
            if part[1] == None:
                continue
            elif type(part[1]) == tuple:
                if not temp_edges:
                    temp_edges.append(part[1])
                else:
                    if part[1] not in temp_edges:
                        temp_edges.append(part[1])
            elif type(part[1]) == list:
                for item in part[1]:
                    if item not in temp_edges:
                        temp_edges.append(item)
        #print("temp_edges = ", temp_edges)
        temp_values = [0] * len(temp_edges)
        for part in temp_velocity:
            if type(part[1]) == tuple:
                #print("tuple index = ", temp_edges.index(part[1]))
                temp_values[temp_edges.index(part[1])] += part[0]
            elif type(part[1]) == list:
                for item in part[1]:
                    #print("list index = ", temp_edges.index(item))
                    temp_values[temp_edges.index(item)] += part[0]
        #print("temp_values = ", temp_values)
        return temp_edges, temp_values

    def add_old_edge_calculate_remove_new_edge(self, old_edge):
        """

        :return:
        """
        # ищем в edges_lines old_edge и добавляем его обратно в self.__local_graph
        #print("Токи в ветвях до добавления ребра: ", old_edge)
        """
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                'type_edge'] != str('Source'):
                print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        """
        for edge in edges_lines:
            if int(edge[0]) == int(old_edge[0]) and int(edge[1]) == int(old_edge[1]):
                self.__local_graph.add_edge(int(edge[0]), int(edge[1]),
                                            resistance=complex(edge[2]),
                                            voltage=complex(edge[3]), type=int(edge[4]),
                                            length=float(edge[5]),
                                            cross_section=float(edge[6]), I=complex(edge[7]),
                                            material=edge[8],
                                            r_0=float(edge[9]), x_0=float(edge[10]),
                                            cos_y=float(edge[11]),
                                            sin_y=float(edge[12]), lose_volt=complex(edge[13]),
                                            lose_energy=complex(edge[14]), PS=str(edge[15]),
                                            type_edge=str(edge[16]), power=complex(edge[17]))
        #algo.func_calculated_current_node_potential_algo_AC(self.__local_graph)
        algo.func_calculated_current_node_potential_algo_AC(self.__local_graph, True)   # для заданных мощностей
        """
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                'type_edge'] != str('Source'):
                print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        """
        #print("Ребро, которое можно удалить = ", self.find_edge(old_edge))
        return self.find_edge(old_edge)

    def find_edge(self, untouchable_edge):
        """

        :return:
        """
        list_edges = []
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                'type_edge'] != str('Source') and list(edge) != untouchable_edge:
                if self.__check_connectivity_graph(edge):
                    list_edges.append([edge, self.__local_graph.edges[edge]['I']])
                    #print(edge, " - можно.")
                else:
                    pass
                    #print(edge, " - нельзя.")
        #print("list_edges = ", list_edges)
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
                    max_limit_for_position += (1 - (cmath.polar(edge[1]).__getitem__(0) / sum_of_currents))
                    correct_list_edges[index].append(1 - (cmath.polar(edge[1]).__getitem__(0) / sum_of_currents))
                    index += 1
                else:
                    sum_of_currents += cmath.polar(edge[1]).__getitem__(0)
        sup_variable = correct_list_edges[0][2]
        roulette = np.random.uniform(0.0, float(max_limit_for_position))
        for item in correct_list_edges:
            if roulette > sup_variable:
                sup_variable += item[2]
            else:
                return item[0]

class Problem(Swarm):
    """
    Класс, который содержит в себе целевую функцию.
    При создании экземпляра данного класса инициализируется весь рой с частицами.
    """
    def __init__(self, graph, swarm_size, constant_K, constant_C1, constant_C2):
        Swarm.__init__(self, graph, swarm_size, constant_K, constant_C1, constant_C2)

    def _final_func(self, local_graph):
        """
        В этой функции определяется целевая функция
        :return:
        """
        # строка ниже и есть целевая функция оптимизации!
        #algo.func_calculated_current_node_potential_algo_AC(local_graph)
        algo.func_calculated_current_node_potential_algo_AC(local_graph, True)  # для заданных мощностей
        final_func = algo.func_law_Joule_Lenz(local_graph, True)
        #final_func = algo.func_loses_energy_400(graph)
        #print("Потери в ветвях для исходной схемы = ", final_func)
        #print("Максимальное значение потерь напряжения (абсолютная величина) = ", abs(graph.nodes[0]['potential']))
        #print("Максимальное значение потерь напряжения в процентах = ", abs(abs((graph.nodes[0]['potential'] / 630) * 100) - 100))
        return final_func

"""
RUN ALGO!
"""

iter_count = 10
swarm_size = 10
#swarm_size = 100
grade_parameters = 2
constant_K = 0.1
constant_C1 = 2
constant_C2 = 3
graph = algo.func_edges_to_undirected_graph_AC(edges, COUNT_NODES)

solve = Problem(graph, swarm_size, constant_K, constant_C1, constant_C2)

for n in range(iter_count):
    #print("Position = ", solve.global_Best_Position)
    #print("Final Func = ", solve.global_Best_Final_func)
    solve.next_iteration()

print("Position = ", solve.global_Best_Position)
print("Final Func = ", cmath.polar(solve.global_Best_Final_func).__getitem__(0))
print("Final Func = ", solve.global_Best_Final_func)

"""
RUN ALGO!
"""

print("Hello, World!")