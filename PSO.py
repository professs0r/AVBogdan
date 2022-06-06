import test_node_potential_algo
from abc import ABCMeta, abstractmethod
import numpy
import numpy.random


"""
#global variables
"""


"""
#global variables
"""
"""
v_id_new = omega*v_id_old + c_1*rand()*(p_id - x_id_old) + c_2*Rand()*(p_gd - x_id_old)
x_id_new = v_id_new + x_id_old
"""

"""
#algo
"""
class Swarm(object):
    """
    Базовый класс для роя частиц. Его нужно переопределять для конкретной целевой функции.
    """
    __metaclass__ = ABCMeta

    def __init__(self, swarmsize, minvalues, maxvalues, currentVelocityRatio,
                localVelocityRatio, globalVelocityRatio):
        """
        swarmsize - размер роя (количество частиц)
        minvalues - список, задающий минимальные значения для каждой координаты частиц
        maxvalues - список, задающий максимальные значения для каждой координаты частицы
        currentVelocityRatio - коэффициент k (общий масштабирующий коэффициент для скорости)
        localVelocityRatio - в формуле переменная c1, коэффициент задающий влияние лучшей точки,
        найденной частицей на будущую скорость
        globalVelocityRatio - в формуле перменная c2, коэффициент задающий влияние лучшей точки,
        найденной всеми частицами на будущую скорость
        """
        self.__swarmsize = swarmsize
        assert len(minvalues) == len(maxvalues)
        assert (localVelocityRatio + globalVelocityRatio) > 4
        self.__minvalues = numpy.array(minvalues[:])
        self.__maxvalues = numpy.array(maxvalues[:])
        self.__currentVelocityRatio = currentVelocityRatio
        self.__localVelocityRatio = localVelocityRatio
        self.__globalVelocityRatio = globalVelocityRatio
        self.__globalBestFinalFunc = None
        self.__globalBestPosition = None
        self.__swarm = self.__createSwarm()

    def __getitem__(self, index):
        """
        Возврщает частицу с заданным номером
        """
        return self.__swarm[index]
    
    def __createSwarm(self):
        """
        Создаёт рой частиц со случайными координатами
        """
        return [Particle(self) for _ in range(self.__swarmsize)]

    def nextIteration(self):
        """
        Выполнить следующую итерацию алгоритма
        """
        for particle in self.__swarm:
            particle.nextIteration(self)

    @property
    def minvalues(self):
        return self.__minvalues

    @property
    def maxvalues(self):
        return self.__maxvalues

    @property
    def currentVelocityRatio(self):
        return self.__currentVelocityRatio

    @property
    def localVelocityRatio(self):
        return self.__localVelocityRatio

    @property
    def globalVelocityRatio(self):
        return self.__globalVelocityRatio

    @property
    def globalBestPosition(self):
        return self.__globalBestPosition

    @property
    def globalBestFinalFunc(self):
        return self.__globalBestFinalFunc

    def getFinalFunc(self, position):
        assert len(position) == len(self.minvalues)
        finalFunc = self._finalFunc(position)
        if(self.__globalBestFinalFunc == None or
                finalFunc < self.__globalBestFinalFunc):
            self.__globalBestFinalFunc = finalFunc
            self.__globalBestPosition = position[:]

    @abstractmethod
    def _finalFunc(self, position):
        pass

    @property
    def dimention(self):
        """
        Возвращает тукущую размерность задачи
        """
        return len(self.minvalues)

    def _getPenalty(self, position, ratio):
        """
        Рассчитать штрафную функцию
        position - координаты, для которых рассчитыватся штраф
        ratio - вес штрафа
        # При отладке, остановился здесь!!
        """
        penalty1 = sum([ratio * abs(coord - minval)
            for coord, minval in zip(position, self.minvalues)
            if coord < minval])
        penalty2 = sum([ratio * abs(coord - maxval)
            for coord, maxval in zip(position, self.maxvalues)
            if coord > maxval])
        return penalty1 + penalty2

class Particle(object):
    """
    Класс, описывающий одну частицу
    """
    def __init__(self, swarm):
        """
        swarm - экземпляр класса Swarm, хранящий параметры алгоритма:
            список частиц, и лучшие значения для роя вцелом
        position - начальное положение частицы (список)
        """
        # Текущее положение частицы
        self.__currentPosition = self.__getInitPosition(swarm)
        # Лучшее положение частицы
        self.__localBestPosition = self.__currentPosition[:]
        # Лучшее значение целевой функции
        self.__localBestFinalFunc = swarm.getFinalFunc(self.__currentPosition)

        self.__velocity = self.__getInitVelocity(swarm)

    @property
    def position(self):
        return self.__currentPosition

    @property
    def velocity(self):
        return self.__velocity

    def __getInitPosition(self, swarm):
        """
        Возвращает список со случайными координатами для заданного интервала изменений
        """
        return numpy.random.rand(swarm.dimention) * (swarm.maxvalues - swarm.minvalues) + swarm.minvalues

    def __getInitVelocity(self, swarm):
        """
        Сгенерировать начальную случайную скорость
        """
        assert len(swarm.minvalues) == len(self.__currentPosition)
        assert len(swarm.maxvalues) == len(self.__currentPosition)
        minval = -(swarm.maxvalues - swarm.minvalues)
        maxval = (swarm.maxvalues - swarm.minvalues)
        return numpy.random.rand(swarm.dimention) * (maxval - minval) + minval

    def nextIteration(self, swarm):
        # Случайный вектор для коррекции скорости с учётом лучшей позиции данной частицы
        rnd_currentBestPosition = numpy.random.rand(swarm.dimention)
        # Случайный вектор для коррекции скорости с учётом лучшей глобальной позиции всех частиц
        rnd_globalBestPosition = numpy.random.rand(swarm.dimention)
        veloRatio = swarm.localVelocityRatio + swarm.globalVelocityRatio
        commonRatio = (2.0 * swarm.currentVelocityRatio /
                (numpy.abs(2.0 - veloRatio - numpy.sqrt(veloRatio**2 - 4.0*veloRatio))))
        # Посчитать новую скорость
        newVelocity_part1 = commonRatio * self.__velocity
        newVelocity_part2 = (commonRatio * swarm.localVelocityRatio * 
                rnd_currentBestPosition * 
                (self.__localBestPosition - self.__currentPosition))
        newVelocity_part3 = (commonRatio * swarm.globalVelocityRatio *
                rnd_globalBestPosition * 
                (swarm.globalBestPosition - self.__currentPosition))
        self.__velocity = newVelocity_part1 + newVelocity_part2 + newVelocity_part3
        # Обновить позицию частицы
        self.__currentPosition += self.__velocity
        finalFunc = swarm.getFinalFunc(self.__currentPosition)
        print(finalFunc)
        if finalFunc < self.__localBestFinalFunc:
            self.__localBestPosition = self.__currentPosition[:]
            self.__localBestFinalFunc = finalFunc

class Swarm_x2(Swarm):
    def __init__(self,
                swarmsize,
                minvalues,
                maxvalues,
                currentVelocityRatio,
                localVelocityRatio,
                globalVelocityRatio):
        Swarm.__init__(self,
                       swarmsize,
                       minvalues,
                       maxvalues,
                       currentVelocityRatio,
                       localVelocityRatio,
                       globalVelocityRatio)

    def _finalFunc(self, position):
        penalty = self._getPenalty(position, 10000.0)
        finalfunc = sum(position * position)
        return finalfunc + penalty

def printResult(swarm, iteration):
    template = u"""Iteration: {iter}
    Best Position: {bestpos}
    Best Final Func: {finalfunc}
    """
    result = template.format(iter=iteration,
            bestpos = swarm.globalBestPosition,
            finalfunc = swarm.globalBestFinalFunc)
    return result

"""
#algo
"""

#if __name__ == "__main__":
iterCount = 300
dimention = 2
#dimention = 5
swarmsize = 2
#swarmsize = 200
minvalues = numpy.array([-100] * dimention)
maxvalues = numpy.array([100] * dimention)
currentVelocityRatio = 0.1
localVelocityRatio = 1.0
globalVelocityRatio = 5.0
swarm = Swarm_x2(swarmsize,
                minvalues,
                maxvalues,
                currentVelocityRatio,
                localVelocityRatio,
                globalVelocityRatio)

for n in range(iterCount):
    print ("Position", swarm[0].position)
    print ("Velocity", swarm[0].velocity)
    print (printResult(swarm, n))
    swarm.nextIteration()


print("Hello, World!")
