import numpy as np
from abc import abstractmethod
import math
import test_node_potential_algo as algo

"""
SUPPORT FUNCTIONS
"""

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
    def __init__(self, graph, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2):
        assert len(min_limit) == len(max_limit), "Размерности ограничений должны быть одинаковыми!"
        assert (constant_C1 + constant_C2) > 4, "Сумма коэффициентов C1 и C2 должна быть больше 4!"
        self.__constant_C1 = constant_C1
        self.__constant_C2 = constant_C2
        self.__constant_K = constant_K
        self.__constant_PHI = constant_C1 + constant_C2
        self.__constant_CHI = self.__get_constant_CHI(self.__constant_PHI, constant_K)
        self.__swarm_size = swarm_size
        self.__min_limit = np.array(min_limit[:])
        self.__max_limit = np.array(max_limit[:])
        self.__global_Best_Final_func = None
        self.__global_Best_position = None
        self.__swarm = self.__create_swarm(graph)

    def __get_constant_CHI(self, constnt_PHI, constant_K):
        result = float(2*constant_K / (abs(2 - constnt_PHI - math.sqrt(constnt_PHI**2 - 4*constnt_PHI))))
        #print("constant_CHI", result)
        return result

    def __create_swarm(self, graph):
        """
        Инициализируем рой
        :return:
        """
        return [Particle(self, graph) for _ in range(self.__swarm_size)]

    def get_final_func(self, graph, position):
        """

        :param position:
        :param graph:
        :return:
        """
        assert len(self.min_limit) == len(position), "Размерность ограничений должна быть равной количеству координат!"
        final_func = self._final_func(graph, position)
        if (self.global_Best_Final_func == None or final_func < self.global_Best_Final_func):
            self.__global_Best_Final_func = final_func
            self.__global_Best_position = position[:]
        #print("final_func", final_func)
        return final_func

    def _get_penalty(self, position, value_penalty):
        """
        Функция для расчёта штрафа за выход частицы за границы поиска   ВОЗМОЖНО, В ДАЛЬНЕЙШЕМ ИМЕЕТ СМЫСЛ ЗАМЕНИТЬ ДАННУЮ ФУНКЦИЮ НА БАНАЛЬНУЮ ПРОВЕРКУ: НЕ ВЫШЛА ЛИ ЧАСТИЦА ЗА ГРАНИЦЫ ПОИСКА
        position - значения (координаты) для которых рассчитывается штраф
        value_penalty - величина штрафа за нарушение границ
        :param position:
        :param value_penalty:
        :return:
        """
        penalty_1 = sum([value_penalty * abs(coord - min_limit) for coord, min_limit in zip(position, self.min_limit)
                         if coord < min_limit])
        penalty_2 = sum([value_penalty * abs(coord - max_limit) for coord, max_limit in zip(position, self.max_limit)
                         if coord > max_limit])
        #print("penalty_1 ", penalty_1)
        #print("penalty_2 ", penalty_2)
        return penalty_1 + penalty_2

    def next_iteration(self, graph):
        """

        :return:
        """
        for particle in self.__swarm:
            particle.update_velocity_and_position(self, graph)

    @abstractmethod
    def _final_func(self, graph, position):
        pass

    @property
    def grade_limit(self):
        """
        функция возвращает количество ограничений
        пример: min_limit - массив из двух знаечний (т.е. есть два ограничения и в этом массиве заданы минимальные
        значения этих ограничений)
        длина массива min_limit = количество ограничений при оптимизации
        :return:
        """
        return len(self.__min_limit)

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
    def min_limit(self):
        return self.__min_limit

    @property
    def max_limit(self):
        return self.__max_limit

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
    def __init__(self, swarm, graph):
        self.__current_position = self.__get_initial_position(graph, swarm)
        self.__velocity = self.__get_initial_velocity(swarm)
        self.__local_Best_position = self.__current_position[:]
        self.__local_Best_Final_func = swarm.get_final_func(graph, self.__current_position)

    @property
    def position(self):
        return self.__current_position

    @property
    def velocity(self):
        return self.__velocity

    def __get_initial_velocity(self, swarm):
        """
        функция возвращает массив значений размерностью равной "grade_limit" в пределах от "min_limit" до "max_limit"
        :param swarm:
        :return:
        """
        assert len(swarm.max_limit) == len(swarm.min_limit), "Размерности ограничений должны быть одинаковыми!"
        assert swarm.grade_limit == len(swarm.min_limit), "Размерность ограничений должна быть равной количеству ограничений!"
        assert len(swarm.min_limit) == len(self.__current_position), "Размерность ограничений должна быть равной количеству координат!"
        min_values = -(swarm.max_limit - swarm.min_limit)
        max_values = (swarm.max_limit - swarm.min_limit)
        result = np.random.rand(swarm.grade_limit) * (max_values - min_values) + min_values
        #print("get_initial_velocity ", result)
        return result

    def __get_initial_position(self, graph, swarm):
        """
        функция возвращает массив значений размерностью равной "grade_limit" в пределах от "min_limit" до "max_limit"
        :param swarm:
        :return:
        """
        assert len(swarm.max_limit) == len(swarm.min_limit), "Размерности ограничений должны быть одинаковыми!"
        assert swarm.grade_limit == len(swarm.min_limit), "Размерность ограничений должна быть равной количеству ограничений!"
        result = np.random.rand(swarm.grade_limit) * (swarm.max_limit - swarm.min_limit) + swarm.min_limit
        result = self.__first_law_Kirchhoff(graph, result)
        #print("get_initial_position ", result)
        return result

    def __first_law_Kirchhoff(self,graph, position):
        """
        функция для проверки адекватности случайно сгенерированных параметров на основе первого закона Кирхгофа;
        :param position: случайно сгенерированные параметры;
        :return: значение параметров, которые согласованы с первым законом Кирхгофа
        """
        # Допустим, что будем отталкиваться от самого первого тока,
        # то есть тока в первой ветви (содержащей источник энергии и наименьшее сопротивление)

        return adequacy_position

    def update_velocity_and_position(self, swarm, graph):
        """

        :param swarm:
        :param graph:
        :return:
        """
        # случайный вектор для коррекции скорости с учётом лучшей позиции данной частицы (rand() - во втором
        # слагаемом уравнения, сразу после константы C1)
        #print("self.__velocity = ", self.__velocity)
        #print("self.__current_position = ", self.__current_position)
        rand_current_Best_Position = np.random.rand(swarm.grade_limit)
        #print("rand_current_Best_Position ", rand_current_Best_Position)
        # случайный вектор для коррекции скорости с учётом лучшей глобальной позиции всех частиц (Rand() - в третьем
        # слагаемом уравнения, сразу после константы C2)
        rand_global_Best_Position = np.random.rand(swarm.grade_limit)
        #print("rand_global_Best_Position ", rand_global_Best_Position)
        # делим выражение для расчёта обновленого значения скорости на отдельные слагаемые
        # первое слагаемое с текущей сокростью частицы
        new_velocity_part_1 = swarm.constant_CHI * self.__velocity
        #print("new_velocity_part_1 ", new_velocity_part_1)
        new_velocity_part_2 = swarm.constant_CHI * (swarm.constant_C1 * rand_current_Best_Position *
                                                    (self.__local_Best_position - self.__current_position))
        #print("new_velocity_part_2 ", new_velocity_part_2)
        new_velocity_part_3 = swarm.constant_CHI * (swarm.constant_C2 * rand_global_Best_Position *
                                                    (swarm.global_Best_Position - self.__current_position))
        #print("new_velocity_part_3 ", new_velocity_part_3)
        self.__velocity = new_velocity_part_1 + new_velocity_part_2 + new_velocity_part_3
        #print("self.__velocity ", self.__velocity)
        # Обновляем позицию частицы
        self.__current_position += self.__velocity
        #print("self.__current_position ", self.__current_position)
        final_func = swarm.get_final_func(graph, self.__current_position)
        #print("final_func", final_func)
        if final_func < self.__local_Best_Final_func:
            self.__local_Best_Final_func = final_func
            self.__local_Best_position = self.__current_position[:]
        #print("self.__local_Best_position = ", self.__local_Best_position)
        #print("self.__local_Best_Final_func = ", self.__local_Best_Final_func)

class Problem(Swarm):
    """
    Класс, который содержит в себе целевую функцию.
    При создании экземпляра данного класса инициализируется весь рой с частицами.
    """
    def __init__(self, graph, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2):
        Swarm.__init__(self, graph, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2)

    # ФУНКЦИЯ НИЖЕ - ФУНКЦИЯ ИЗ ИСХОДНИКОВ АЛГОРИТМА ОПТИМИЗАЦИИ, ЗАКОММЕНТИРОВАЛ ДЛЯ ТОГО, ЧТОБЫ
    # ВНЕДРИТЬ СОБСТВЕННУЮ ЦЕЛЕВУЮ ФУНКЦИЮ (ДЛЯ ОПТИМИЗАЦИИ) НА ОСНОВЕ ЗНАЧЕНИЙ ПОТЕРЬ НАПРЯЖЕНИЯ В ЭЛЕКТРИЧЕСКОЙ СХЕМЕ
    #def _final_func(self, position):
        """
        В этой функции определяется целевая функция
        :return:
        """
    #    penalty = self._get_penalty(position, 10000.0)
        # строка ниже и есть целевая функция оптимизации!
    #    final_func = sum(position * position)
        #print("final_func ", final_func)
    #    return final_func + penalty
    def _final_func(self, graph, position):
        """
        В этой функции определяется целевая функция
        :param position:
        :return:
        """
        penalty = self._get_penalty(position, 10000.0)
        final_func = 0
        T_p = 1
        index = 0
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['lose_energy'] = (3 * algo.k_konf * algo.k_form * T_p * graph[edge[0]][edge[1]]['r_0'] *
                                                      graph[edge[0]][edge[1]]['length'] *
                                                      pow(position[index], 2)) / (1000)
            final_func += graph[edge[0]][edge[1]]['lose_energy']
            index += 1
        print("final_func = ", final_func)
        return final_func + penalty

"""
RUN ALGO!
"""

def func_run_algo(graph, count_iter, size, k, c1, c2, value_min_limit, value_max_limit):
    #iter_count = 100
    iter_count = count_iter
    #swarm_size = 10
    swarm_size = size
    # grade_parameters = 2
    #grade_parameters = 17  # для проверки работоспособности алгоритма роя частиц на схеме с 10 узлами и 17 рёбрами
    # ЗДЕСЬ НЕОБХОДИМО НАПИСАТЬ ИНСТРУКЦИЮ ДЛЯ ВЫЧИСЛЕНИЕ КОЛИЧЕСТВА ПАРАМЕТРОВ, ПО КОТОРЫМ БУДЕТ ПРОИЗВОДИТЬСЯ
    # ОПТИМИЗАЦИЯ
    grade_parameters = len(graph.edges)
    #constant_K = 0.1
    constant_K = k
    #constant_C1 = 2
    constant_C1 = c1
    #constant_C2 = 3
    constant_C2 = c2
    # min_limit = numpy.array([-100] * grade_parameters)
    # строка кода ниже для проверки работоспособности алгоритма роя частиц на схеме с 10 узлами и 17 рёбрами
    # min_limit = numpy.array([49.39, 22.41, 26.98, 10.49, 11.92, 8.161, 2.324, 11, 15.98, 14.37, 3.323, 5.23, 5.647, 14.28, 1.695, 6.925, 12.57])
    min_limit = np.array([value_min_limit] * grade_parameters)
    # max_limit = numpy.array([100] * grade_parameters)
    max_limit = np.array([value_max_limit] * grade_parameters)  # для проверки работоспособности алгоритма роя частиц на схеме с 10 узлами и 17 рёбрами

    solve = Problem(graph, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2)

    for n in range(iter_count):
        print("global_Best_Position = ", solve.global_Best_Position)
        print("global_Best_Final_func = ", solve.global_Best_Final_func)
        solve.next_iteration(graph)

"""
RUN ALGO!
"""

#                                        START SOURCE DATA
#                                        START SOURCE DATA
#                                        START SOURCE DATA

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

# edge_0 = (source=0, finish=1, resistance=70.1, voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated))

edge_0 = (0, 1, 0.1, 630, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_1 = (1, 2, 1.17, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_2 = (1, 4, 1.35, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_3 = (2, 3, 2.55, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_4 = (2, 5, 2.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_5 = (3, 0, 70.1, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_6 = (3, 6, 1.68, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_7 = (4, 5, 1.25, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_8 = (4, 7, 1.08, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_9 = (5, 0, 40, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_10 = (5, 6, 2.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_11 = (5, 8, 1.25, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_12 = (6, 9, 2.44, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_13 = (7, 0, 40, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_14 = (7, 8, 1.79, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_15 = (8, 9, 2.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_16 = (9, 0, 44.1, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)

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
                  edge_16])

#                                        START SOURCE DATA
#                                        START SOURCE DATA
#                                        START SOURCE DATA

#                                        START WORKING ALGO
#                                        START WORKING ALGO
#                                        START WORKING ALGO


matrix = algo.func_list_to_matrix(directed_adjacency_list)
nodes = algo.func_count_of_nodes(matrix)
branches = algo.func_count_of_branches(directed_adjacency_list)
graph = algo.func_edges_to_directed_graph(edges, COUNT_NODES)
algo.func_calculating_support_variables(graph)
algo.func_calculated_current_node_potential_algo(graph, nodes, nodes - 1, matrix, directed_adjacency_list)
print("Величины токов в исходном графе: ")
for branch in graph.edges():
    print(graph.edges[branch]['I'])
count_iter = 1
size = 3
k = 0.1
c1 = 2
c2 = 3
value_min_limit = -500
value_max_limit = 500
func_run_algo(graph, count_iter, size, k, c1, c2, value_min_limit, value_max_limit)

#                                        END WORKING ALGO
#                                        END WORKING ALGO
#                                        END WORKING ALGO

print("\n\nЕсли Ты видишь это сообщение, значит программа отработала корректно!\n\n")