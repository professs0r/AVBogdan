import math

import networkx as nx
import numpy as np
from abc import abstractmethod
import math
import test_node_potential_algo as algo
import spanning_tree as st
import copy

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
        for i in new_vertexes:
            lens[i] = lens[next] + 1
        to_explore.extend(new_vertexes)
        visited.update(new_vertexes)
    return lens

def is_connected(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "Connectivity is undefined ", "for the null graph."
        )
    result = sum(1 for node in func_BFS(G, 0)) == len(G)
    return result

"""
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
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
# для построения всех остовных деревьев
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
                  edge_12])
"""

"""
graph = algo.func_edges_to_undirected_graph(edges, COUNT_NODES)

# подсчёт и перечисление всех остовных деревьев
trees = st.func_networkx_build_spanning_tree(graph)
list_edges_in_spanning_tree = []
lowest_loses = 10000000000000000000000000000000000000000000000000000
for tree in trees:
    for edge in tree.edges.items():
        list_edges_in_spanning_tree.append(edge[0])
    new_graph = algo.func_list_of_edges_to_graph(list_edges_in_spanning_tree, COUNT_NODES, edges_lines, edges_nagr)
    algo.func_calculated_current_node_potential_algo(new_graph)
    #for branch in new_graph.edges():
        #print(branch, new_graph.edges[branch]['I'])
    loses = 0.0
    for edge in new_graph.edges():
        if str(new_graph.edges[edge]['type_edge']) == str('Branch'):
            temp_variable = new_graph.edges[edge]['resistance'] * abs(pow(new_graph.edges[edge]['I'], 2))
            loses += temp_variable
    if loses < lowest_loses:
        lowest_loses = loses
        print("for tree:", list_edges_in_spanning_tree, "loses = ", lowest_loses)
    new_graph.clear()
    list_edges_in_spanning_tree.clear()
print("Проверка!")
"""

"""
algo.func_calculated_current_node_potential_algo(graph)
print("Величины токов в исходном графе: ")
for branch in graph.edges():
    print(branch)
    print(graph.edges[branch]['I'])
"""

"""
# подсчёт и перечисление всех остовных деревьев
count_trees = algo.func_Kirchhoff(graph)
print("Количество остовных деревьев по теореме Кирхгоффа: ", count_trees)
trees = st.func_networkx_build_spanning_tree(graph)
print("Количество остовных деревьев по алгоритму перечисления последних: ", len(trees))
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
        self.__swarm_size = swarm_size
        self.__global_Best_Final_func = None
        self.__global_Best_position = None
        self.__swarm = self.__create_swarm()

    def __get_constant_CHI(self, constnt_PHI, constant_K):
        return float(2*constant_K / (abs(2 - constnt_PHI - math.sqrt(constnt_PHI**2 - 4*constnt_PHI))))

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
        final_func = self._final_func(local_graph)
        if (self.global_Best_Final_func == None or final_func < self.global_Best_Final_func):
            self.__global_Best_Final_func = final_func
            self.__global_Best_position = position
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

class Particle(Swarm):
    """
    Класс, описывающий одну частицу
    """
    def __init__(self, swarm):
        self.__local_graph = graph.copy()
        #edge_for_remove = (4,7)
        #self.__test_connectivity_graph(edge_for_remove)
        self.__count_edges= self.__count_edges_for_remove()
        #print("self.__count_edges_for_remove = ", self.__count_edges_for_remove)
        self.__grade_of_change = 0.0
        if self.__check_graph():
            self.__current_position = self.__get_initial_position(swarm)
            self.__velocity = self.__get_initial_velocity(swarm)
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

    def __test_connectivity_graph(self, edge_for_remove):
        """

        :param edge_for_remove:
        :return:
        """
        print(self.__local_graph.edges[edge_for_remove])
        algo.func_calculated_current_node_potential_algo(self.__local_graph)
        print("Токи в графе:")
        for edge in self.__local_graph.edges():
            print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        print("self.__local_graph.edges[edge_for_remove] = ",self.__local_graph.edges[edge_for_remove])
        temp_edge = list(self.__local_graph.edges[edge_for_remove].values())
        print("temp_edge = ", temp_edge)
        self.__local_graph.remove_edge(edge_for_remove[0], edge_for_remove[1])
        algo.func_calculated_current_node_potential_algo(self.__local_graph)
        print("Токи в графе после удаления ребра:")
        for edge in self.__local_graph.edges():
            print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        print("temp_edge = ", temp_edge)
        print("Пытаемся добавить ребро обратно: ")
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
        algo.func_calculated_current_node_potential_algo(self.__local_graph)
        print("Токи в графе после возвращения ребра в граф:")
        for edge in self.__local_graph.edges():
            print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        print("Тесто-блок!")
        print("Удаляю поочерёдно следующие рёбра: 4-7, 5-8, 6-9")
        print("После каждого удаления произвожу расчёт параметров цепи (савниваю их с Matlab) и проверяю граф на связность")
        testo_list = [[4,7], [5,8], [6,9]]
        for rem_edge in testo_list:
            print("Удаляю ребро ", rem_edge, " :")
            self.__local_graph.remove_edge(rem_edge[0], rem_edge[1])
            algo.func_calculated_current_node_potential_algo(self.__local_graph)
            print("Токи в графе после удаления ребра ", rem_edge, " : ")
            for edge in self.__local_graph.edges():
                print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
            print("Начинаем проверку на связность.")
            if is_connected(self.__local_graph):
                print("Проверка на связность успешно пройдена!!!")
            else:
                print("Проверка на связность не пройдёна. Связность нарушена!!!")
            print("Итерация!")
            """
            graph.add_edge(int(temp_edges[iter][0]), int(temp_edges[iter][1]), resistance=float(temp_edges[iter][2]),
                           voltage=float(temp_edges[iter][3]), type=int(temp_edges[iter][4]),
                           length=float(temp_edges[iter][5]),
                           cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]),
                           material=temp_edges[iter][8],
                           r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]),
                           cos_y=float(temp_edges[iter][11]),
                           sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                           lose_energy=float(temp_edges[iter][14]), PS=str(temp_edges[iter][15]),
                           type_edge=str(temp_edges[iter][16]))
            """
        #return True
        #return False
    """
    # старая версия функции
    def __check_connectivity_graph(self, edge_for_remove):
        #функция для проверки связности графа
        nodes = set()
        for edge in self.__local_graph.edges():
            if str(self.__local_graph.edges[edge]['type_edge']) == str('Branch') and list(edge) != list(edge_for_remove):
                nodes.add(int(edge[0]))
                nodes.add(int(edge[1]))
        if list(nodes) == list(range(1, int(self.__local_graph.number_of_nodes()))):
            return True
        return False
    """

    # новая версия функции
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

    def __list_connectivity_graph(self, list_of_edges):
        """
        список рёбер, которые можно удалить из графа и связность не нарушится
        :param list_of_edges:
        :return:
        """
        for edge in list_of_edges:
            if not self.__check_connectivity_graph(edge[0]):
            #if not self.__test_connectivity_graph(edge):
                list_of_edges.remove(edge)
        return list_of_edges

    def __check_graph(self):
        """
        функция для проверки возможности удаления ребра из графа
        :return:
        """
        counter_of_branches = 0
        for edge in self.__local_graph.edges():
            if str(self.__local_graph.edges[edge]['type_edge']) == str('Branch'):
                counter_of_branches += 1
        if counter_of_branches == int(self.__local_graph.number_of_nodes() - 1):
            return False
        return True

    def __max_edges_for_remove(self):
        """
        количество рёбер, которые не являются нагрузками (т.е. ветви, хорды и с источниками)
        :return:
        """
        list_edges = []
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load'):
                # блок без проверки на связность
                list_edges.append(edge)
                """
                if self.__check_connectivity_graph(edge):
                    # блок с проверкой на связность
                    list_edges.append(edge)
                """
        return list_edges

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
        #return np.random.uniform(-(self.__grade_of_change),self.__grade_of_change)
        return np.random.rand()

    def __get_initial_position(self, swarm):
        """
        везде меняю graph на self.__local_graph
        :param swarm:
        :return:
        """
        position = []
        while self.__count_edges:
            algo.func_calculated_current_node_potential_algo(self.__local_graph)
            list_edges = []
            for edge in self.__local_graph.edges():
                if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                    'type_edge'] != str('Source'):
                    if self.__check_connectivity_graph(edge):
                        list_edges.append([edge, self.__local_graph.edges[edge]['I']])
            sort_array = algo.quick_sort(list_edges)
            print("Отсортированный массив : ", '\n', sort_array)
            for i in sort_array:
                print("Ребро: ", i[0], "ток = ", i[1])
            number_of_excess_edges = int(len(sort_array) / 4)
            for num in range(number_of_excess_edges):
                sort_array.pop()
            del number_of_excess_edges
            print("Отсортированный список после удаления рёбер с наибольшими токами:")
            for i in sort_array:
                print("Ребро: ", i[0], "ток = ", i[1])
            correct_list_edges = self.__list_connectivity_graph(sort_array)
            print("Список после проверки на возможность удаления рёбер без нарушения связности:")
            for i in correct_list_edges:
                print("Ребро: ", i[0], "ток = ", i[1])
            sum_of_currents = 0.0
            max_limit_for_position = 0
            for iter in range(2):
                index = 0
                for edge in sort_array:
                    if iter:
                        max_limit_for_position += (1 - abs(edge[1] / sum_of_currents))
                        sort_array[index].append(1 - abs(edge[1] / sum_of_currents))
                        index += 1
                    else:
                        sum_of_currents += abs(edge[1])
            print("max_limit_for_position = ", max_limit_for_position)
            print("sum_of_currents = ", sum_of_currents)
            print("Список с процентами:")
            for i in sort_array:
                print("Ребро: ", i[0], "ток = ", i[1], "процент = ", i[2])
            sup_variable = sort_array[0][2]
            roulette = np.random.uniform(0.0, float(max_limit_for_position))
            for item in sort_array:
                if roulette > sup_variable:
                    sup_variable += item[2]
                    #continue
                else:
                    print("Удаляем ребро: ", item[0])
                    position.append(item[0])
                    self.__local_graph.remove_edge(item[0][0], item[0][1])
                    self.__count_edges -= 1
                    """
                    algo.func_calculated_current_node_potential_algo(self.__local_graph)
                    for edge in self.__local_graph.edges():
                        print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
                    """
                    break

            print("Отладка!")

        print("position = ", position)
        algo.func_calculated_current_node_potential_algo(self.__local_graph)
        print("Токи в ветвях после удаления хорд перечисленных выше.")
        for edge in self.__local_graph.edges():
            if self.__local_graph.edges[edge]['type_edge'] != str('Load') and self.__local_graph.edges[edge][
                'type_edge'] != str('Source'):
                print("Ребро", edge, " Ток = ", self.__local_graph.edges[edge]['I'])
        return position

    def update_velocity_and_position(self, swarm):
        """
        :param swarm:
        :return:
        """
        # случайный вектор для коррекции скорости с учётом лучшей позиции данной частицы (rand() - во втором
        # слагаемом уравнения, сразу после константы C1)
        rand_current_Best_Position = np.random.random()
        # случайный вектор для коррекции скорости с учётом лучшей глобальной позиции всех частиц (Rand() - в третьем
        # слагаемом уравнения, сразу после константы C2)
        rand_global_Best_Position = np.random.random()
        # делим выражение для расчёта обновленого значения скорости на отдельные слагаемые
        # первое слагаемое с текущей сокростью частицы
        new_velocity_part_1 = swarm.constant_CHI * self.__velocity
        new_velocity_part_2 = swarm.constant_CHI * (swarm.constant_C1 * rand_current_Best_Position *
                                                    (self.__local_Best_position - self.__current_position))
        new_velocity_part_3 = swarm.constant_CHI * (swarm.constant_C2 * rand_global_Best_Position *
                                                    (swarm.global_Best_Position - self.__current_position))
        self.__velocity = new_velocity_part_1 + new_velocity_part_2 + new_velocity_part_3
        # Обновляем позицию частицы
        if (self.__current_position + self.__velocity) > self.__grade_of_change:
            above_range = (self.__current_position + self.__velocity) - self.__grade_of_change
            self.__current_position = self.__grade_of_change - above_range
        elif (self.__current_position + self.__velocity) < 0:
            self.__current_position = abs(self.__current_position + self.__velocity)
        else:
            self.__current_position += self.__velocity
        final_func = swarm.get_final_func(self.__current_position)
        #print("self.__local_Best_position", self.__local_Best_position)
        #print("self.__current_position", self.__current_position)
        #print("self.__local_Best_Final_func", self.__local_Best_Final_func)
        #print("final_func", final_func)
        if final_func < self.__local_Best_Final_func:
            self.__local_Best_position = self.__current_position
            self.__local_Best_Final_func = final_func

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
        algo.func_calculated_current_node_potential_algo(local_graph)
        final_func = algo.func_law_Joule_Lenz(local_graph)
        #final_func = algo.func_loses_energy_400(graph)
        #print("Потери в ветвях для исходной схемы = ", final_func)
        #print("Максимальное значение потерь напряжения (абсолютная величина) = ", abs(graph.nodes[0]['potential']))
        #print("Максимальное значение потерь напряжения в процентах = ", abs(abs((graph.nodes[0]['potential'] / 630) * 100) - 100))
        return final_func

"""
RUN ALGO!
"""

iter_count = 3
swarm_size = 3
#swarm_size = 100
grade_parameters = 2
constant_K = 0.1
constant_C1 = 2
constant_C2 = 3
graph = algo.func_edges_to_undirected_graph(edges, COUNT_NODES)

solve = Problem(graph, swarm_size, constant_K, constant_C1, constant_C2)

for n in range(iter_count):
    print("Position = ", solve.global_Best_Position)
    print("Final Func = ", solve.global_Best_Final_func)
    solve.next_iteration()

"""
RUN ALGO!
"""

print("Hello, World!")