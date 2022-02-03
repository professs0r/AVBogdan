# math culculation optimization algorithm

import numpy
from numpy import linalg
import math
import networkx


# global variables

COUNT_NODES = 10
COUNT_BRANCHES = 17

# инициализация графа
def func_initializetion(count_nodes, count_branches):
    graph = 
    return graph

# функция для добавления рёбер
def func_add_edge(source, goal, graph=None):
    graph.add_edge(source, goal)
    graph.add_edge(goal, source)

array_of_resistance = numpy.zeros((COUNT_BRANCHES, COUNT_BRANCHES))
array_of_EMF = numpy.zeros((COUNT_BRANCHES, 1))

array_of_resistance = ([[0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,1],
                       [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],
                       [0,-1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,1,-1,0,0,0,0,0,0,1,0,0,0],
                       [-1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
                       [0,0,-1,-1,1,0,0,0,1,0,0,0,1,0,0,0,0],
                       [0,0,0,0,-1,-1,0,0,0,0,1,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,-1,0,1,0,0,0,0,1,0,0],
                       [0,0,0,0,0,0,0,0,-1,-1,0,1,0,0,0,0,0],
                       [1.35,-1.17,-2.01,1.25,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,2.01,0,2.01,-1.68,-2.55,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,-1.25,0,0,0,1.08,-1.25,1.79,0,0,0,0,0,0,0],
                       [0,0,0,0,-2.01,0,0,0,1.25,0,-2.44,2.01,0,0,0,0,0],
                       [0,1.17,0,0,0,0,2.55,0,0,0,0,0,0,70.1,0,0,0.1],
                       [0,0,0,0,0,0,0,0,0,-1.79,0,-2.01,0,0,40,-44.1,0],
                       [-1.35,0,0,0,0,0,0,-1.08,0,0,0,0,0,0,-40,0,-0.1],
                       [-1.35,0,0,-1.25,0,0,0,0,0,0,0,0,-40,0,0,0,-0.1]]
                       )

array_of_EMF = ([[0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [0],
                 [630],
                 [0],
                 [-630],
                 [-630]])


# строки кода ниже решают системы уравнений (можно пользоваться любой из них результат одинаковый)
#I = numpy.linalg.inv(R).dot(E)
#I = numpy.linalg.solve(array_of_resistance,array_of_EMF)

# создаём экземпляр класса Граф
graph = func_initialization(COUNT_NODES, COUNT_BRANCHES)

for nodes in range(COUNT_NODES):
    graph.add_node(nodes)

print(graph)
