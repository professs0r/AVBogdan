import math

import numpy
from abc import abstractmethod
import math

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
    def __init__(self, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2):
        assert len(min_limit) == len(max_limit), "Размерности ограничений должны быть одинаковыми!"
        assert (constant_C1 + constant_C2) > 4, "Сумма коэффициентов C1 и C2 должна быть больше 4!"
        self.__constant_C1 = constant_C1
        self.__constant_C2 = constant_C2
        self.__constant_K = constant_K
        self.__constant_PHI = constant_C1 + constant_C2
        self.__constant_CHI = self.__get_constant_CHI(self.__constant_PHI, constant_K)
        self.__swarm_size = swarm_size
        self.__min_limit = numpy.array(min_limit[:])
        self.__max_limit = numpy.array(max_limit[:])
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

    def get_final_func(self, position):
        """

        :param position:
        :return:
        """
        assert len(self.min_limit) == len(position), "Размерность ограничений должна быть равной количеству координат!"
        final_func = self._final_func(position)
        if (self.global_Best_Final_func == None or final_func < self.global_Best_Final_func):
            self.__global_Best_Final_func = final_func
            self.__global_Best_position = position[:]
        return final_func

    def _get_penalty(self, position, value_penalty):
        """
        Функция для расчёта штрафа за выход частицы за границы поиска
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
        return penalty_1 + penalty_2

    def next_iteration(self):
        """

        :return:
        """
        for particle in self.__swarm:
            particle.update_velocity_and_position(self)

    @abstractmethod
    def _final_func(self, position):
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
    def __init__(self, swarm):
        self.__current_position = self.__get_initial_position(swarm)
        self.__velocity = self.__get_initial_velocity(swarm)
        self.__local_Best_position = self.__current_position[:]
        self.__local_Best_Final_func = swarm.get_final_func(self.__current_position)

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
        return numpy.random.rand(swarm.grade_limit) * (max_values - min_values) + min_values

    def __get_initial_position(self, swarm):
        """
        функция возвращает массив значений размерностью равной "grade_limit" в пределах от "min_limit" до "max_limit"
        :param swarm:
        :return:
        """
        assert len(swarm.max_limit) == len(swarm.min_limit), "Размерности ограничений должны быть одинаковыми!"
        assert swarm.grade_limit == len(swarm.min_limit), "Размерность ограничений должна быть равной количеству ограничений!"
        return numpy.random.rand(swarm.grade_limit) * (swarm.max_limit - swarm.min_limit) + swarm.min_limit

    def update_velocity_and_position(self, swarm):
        """

        :param swarm:
        :return:
        """
        # случайный вектор для коррекции скорости с учётом лучшей позиции данной частицы (rand() - во втором
        # слагаемом уравнения, сразу после константы C1)
        rand_current_Best_Position = numpy.random.rand(swarm.grade_limit)
        # случайный вектор для коррекции скорости с учётом лучшей глобальной позиции всех частиц (Rand() - в третьем
        # слагаемом уравнения, сразу после константы C2)
        rand_global_Best_Position = numpy.random.rand(swarm.grade_limit)
        # делим выражение для расчёта обновленого значения скорости на отдельные слагаемые
        # первое слагаемое с текущей сокростью частицы
        new_velocity_part_1 = swarm.constant_CHI * self.__velocity
        new_velocity_part_2 = swarm.constant_CHI * (swarm.constant_C1 * rand_current_Best_Position *
                                                    (self.__local_Best_position - self.__current_position))
        new_velocity_part_3 = swarm.constant_CHI * (swarm.constant_C2 * rand_global_Best_Position *
                                                    (swarm.global_Best_Position - self.__current_position))
        self.__velocity = new_velocity_part_1 + new_velocity_part_2 + new_velocity_part_3
        # Обновляем позицию частицы
        self.__current_position += self.__velocity
        final_func = swarm.get_final_func(self.__current_position)
        print("self.__local_Best_position", self.__local_Best_position)
        print("self.__current_position", self.__current_position)
        print("self.__local_Best_Final_func", self.__local_Best_Final_func)
        print("final_func", final_func)
        if final_func < self.__local_Best_Final_func:
            self.__local_Best_position = self.__current_position[:]
            self.__local_Best_Final_func = final_func

class Problem(Swarm):
    """
    Класс, который содержит в себе целевую функцию.
    При создании экземпляра данного класса инициализируется весь рой с частицами.
    """
    def __init__(self, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2):
        Swarm.__init__(self, swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2)

    def _final_func(self, position):
        """
        В этой функции определяется целевая функция
        :return:
        """
        penalty = self._get_penalty(position, 10000.0)
        # строка ниже и есть целевая функция оптимизации!
        final_func = sum(position * position)
        return final_func + penalty

"""
RUN ALGO!
"""

iter_count = 300
swarm_size = 200
grade_parameters = 2
constant_K = 0.1
constant_C1 = 2
constant_C2 = 3
min_limit = numpy.array([-100] * grade_parameters)
max_limit = numpy.array([100] * grade_parameters)

solve = Problem(swarm_size, min_limit, max_limit, constant_K, constant_C1, constant_C2)

for n in range(iter_count):
    print("Position = ", solve.global_Best_Position)
    print("Velocity = ", solve.global_Best_Final_func)
    solve.next_iteration()

"""
RUN ALGO!
"""

print("Hello, World!")