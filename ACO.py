import numpy as np  # просто потому что нужен нумпай
import math  # для математичесих операций
import random as rn  # для генерирования случайных значений

alpha = 1.0  # константа, определяющая жадность алгоритма (выбирается опытным путём)
beta = 1.0  # константа, определяющая стадность алгоритма (выбирается опытным путём)
pheromone = 0.2  # количество феромона между двумя точками (начальное значение)
const_distance_Q = 1.6  # константа, которая находится в числителе (выбирается опытным путём)
MAX_POINTS = 70  # максимальное количество точек для прохождения
# тестовая глобальная переменная для максимального количества муравьёв
MAX_ANTS = 1000  # максимальное количество муравьёв
# предполагаемая рабочая глобальная переменная для максимального количества муравьёв
#MAX_ANTS = 18  # максимальное количество муравьёв
MAX_DISTANCE = 100  # максимальное расстояние (скорее всего, между точками) (пока не совсем понимаю, для чего нужна эта переменная)
MAX_LENGTH = MAX_POINTS * MAX_DISTANCE  # максимальная длина пути (наверно) (пока не совсем понимаю, для чего нужна эта переменная)
# тестовая глобальная переменная для максимального количества итераций
MAX_ITERATIONS = 3000  # максимальное количество итераций, после которого алгоритм останавливается
# предполанаемая рабочая глобальная переменная для максимального количества итераций
#MAX_ITERATIONS = MAX_POINTS * MAX_DISTANCE  # максимальное количество итераций, после которого алгоритм останавливается
p = 0.36  # интенсивность испарения
rho = (1 - p)  # коэффициент испарения
best_length = 1000000000  # переменная для записи самого короткого пути
best_index = 0  # переменная для записи лучшего индекса (возможно муравья)


# определяем класс (Муравей)
class ANT:
    # инициализация
    def __init__(self):
        self.current_point = 0
        self.next_point = -1
        self.visited_points = []
        self.current_path = []
        self.distance = 0.0
        self.index_of_path = 1
        pass

    pass

    # функция для расчёта вероятности перехода муравья из одной точки в другую
    def func_probability_to_move(self, matrix_of_distance_after_division, pheromone_matrix):
        # матрица для хранения значений произведений расстояния на количество ферромона
        multy_length_and_pheromone = np.zeros(len(points))
        # временная переменная, в которую записывается сумма всех желаний перейти из i-ой точки в j-ую
        sum_of_all_wishes = 0
        start = self.current_point
        for stop in range(MAX_POINTS):
            if (self.visited_points[stop] == 0):
                multy_length_and_pheromone[stop] = pow(pheromone_matrix[start][stop], alpha) * pow(matrix_of_distance_after_division[start][stop], beta)
                sum_of_all_wishes += multy_length_and_pheromone[stop]
        stop = 0
        matrix_probability_to_move = np.zeros(len(multy_length_and_pheromone))
        for stop in range(len(multy_length_and_pheromone)):
            matrix_probability_to_move[stop] = multy_length_and_pheromone[stop] / sum_of_all_wishes
        return matrix_probability_to_move

    # функция для определения следующей точки для перехода муравья
    def func_choice_next_point(self, matrix_probability_to_move):
        # генерируем случайно число от 0 до 1, для выбора следующей точки для перехода
        supp_rand_num = rn.uniform(0.0000000000000001, 1.0)
        supp_matrix = np.zeros(len(matrix_probability_to_move))
        sum_of_probability = matrix_probability_to_move[0]
        for index in range(len(matrix_probability_to_move)):
            if sum_of_probability < supp_rand_num:
                sum_of_probability += matrix_probability_to_move[index + 1]
            else:
                return index

    # функция для отображения значений перменных
    def func_print_info(self):
        print("next_point = ", self.next_point)
        print("current_point = ", self.current_point)
        print("current_path = ", self.current_path)
        print("visited_points = ", self.visited_points)
        print("index_of_path = ", self.index_of_path)
        print("distance = ", self.distance)


# определяем класс (Точка)
class POINT:
    # инициализация
    def __init__(self, x, y, nominal=0):
        self.x_coordinate = x
        self.y_coordinate = y
        self.array_of_coordinates = np.array([x, y])
        self.nominal = nominal
        pass
    pass

    # функция для отображения значений перменных
    def func_print_info(self):
        print("x_coordinate = ", self.x_coordinate)
        print("y_coordinate = ", self.y_coordinate)
        print("array_of_coordinates = ", self.array_of_coordinates)
        print("nominal = ", self.nominal)

# считаем расстояния между всеми точками
def func_set_distance_about_all_points(coordinates):
    count_of_points = len(coordinates)
    matrix_of_distance = np.zeros((count_of_points, count_of_points))
    for i in range(count_of_points):
        for j in range(i, count_of_points):
            # np.linalg.norm заменяет длинную формулу метрики Евклида
            matrix_of_distance[i][j] = matrix_of_distance[j][i] = np.linalg.norm(
                coordinates[i].array_of_coordinates - coordinates[j].array_of_coordinates)
    return matrix_of_distance


# функция для генерирования матрицы, отображающей информацию о ферромонах на гранях
def func_set_pheromone_matrix(coordinates):
    count_of_points = len(coordinates)
    pheromone_matrix = np.zeros((count_of_points, count_of_points))
    # изначальное количество феромонов на гранях (на нулевой итерации)
    pheromone_matrix += pheromone
    return pheromone_matrix

# функция для моделирования перехода муравья из одной точки в другую
def func_modeling_of_move(ants):
    success_to_move = 0
    for stop in range(MAX_ANTS):
        if ants[stop].index_of_path < MAX_POINTS:
            # создаём матрицу, в которую записываются вероятности переходов из i-ой точки в j-ую
            matrix_probability_to_move = ants[stop].func_probability_to_move(matrix_of_distance_after_division, pheromone_matrix)
            ants[stop].next_point = ants[stop].func_choice_next_point(matrix_probability_to_move)
            ants[stop].visited_points[ants[stop].next_point] = 1
            ants[stop].current_path[ants[stop].index_of_path] = ants[stop].next_point
            ants[stop].index_of_path += 1
            ants[stop].distance += matrix_of_distance_before_division[ants[stop].current_point][ants[stop].next_point]
            # в исходном коде данный блок исполняется, я же попробую его закомментировать, так как, по идее, переход к первоначальной точке необязателен
            # возможно, из-за этого возникнут проблемы при запуске функции func_update_pheromone в блоке нанесения нового количества ферромонов
            #if ants[stop].index_of_path == MAX_POINTS:
                # в переменную distance прибавляется значение расстояния между последней точкой в пути муравья и исходной точкой маршрута муравья
                #ants[stop].distance += matrix_of_distance_before_division[ants[stop].current_path[MAX_POINTS - 1]][ants[stop].current_path[0]]
            ants[stop].current_point = ants[stop].next_point
            success_to_move += 1
    return success_to_move

# функция, которая пересчитывает количество феромонов на гранях
def func_update_pheromone(ants):
    start = 0
    stop = 0
    # испарение фермента
    for start in range(MAX_POINTS):
        for stop in range(MAX_POINTS):
            if start != stop:
                pheromone_matrix[start][stop] *= rho
                if pheromone_matrix[start][stop] < 0.0:
                    pheromone_matrix[start][stop] = pheromone
    #добавление нового количества ферментов
    for ant in range(MAX_ANTS):
        for point in range(MAX_POINTS):
            if point < MAX_POINTS - 1:
                start = ants[ant].current_path[point]
                stop = ants[ant].current_path[point + 1]
            else:
                #print("Закомментированный блок в функции func_update_pheromone!")
                break
                # я так думаю, что нужно добавить брейк
                # попробовал закомментировать, так как в функции func_modeling_of_move, так же закомментировал один блок
                #start = self.current_path[point]
                #stop = self.current_path[0]
            pheromone_matrix[start][stop] += const_distance_Q / ants[ant].distance
            pheromone_matrix[stop][start] = pheromone_matrix[start][stop]
    # добавил на всякий случай, чтобы обнулить значения переменных
    start = 0
    stop = 0
    for start in range(MAX_POINTS):
        for stop in range(MAX_POINTS):
            pheromone_matrix[start][stop] *= p

# функция для сброса муравьёв
def func_refresh(ants):
    global best_length
    for ant in range(MAX_ANTS):
        # была переменная "best", я думаю, что это ошибка поэтому исправил на "best_length"
        if (ants[ant].distance < best_length):
            # была переменная "best", я думаю, что это ошибка поэтому исправил на "best_length"
            best_length = ants[ant].distance
            best_index = i
        ants[ant].next_point = -1
        ants[ant].distance = 0.0
        for index in range(MAX_POINTS):
            # небходимо не добавлять нули, а заменять существующие единицы нулями
            ants[ant].visited_points[index] = 0
            # необходимо не добавлять -1, а заменять существующие значения точек
            ants[ant].current_path[index] = -1
        ants[ant].current_point = 0
        ants[ant].index_of_path = 1
        ants[ant].current_path[0] = ants[ant].current_point
        ants[ant].visited_points[ants[ant].current_point] = 1


############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
####################################                    ЗАПУСКАЕМ АЛГОРИТМ                ##################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

number_of_current_iteration = 0

# создаём перечень объектов "Точка"
p_source = POINT(4.5, 7.7)
point_1 = POINT(1.1, 5.2)
point_2 = POINT(4.6, 2.7)
point_3 = POINT(7.2, 3.8)
point_4 = POINT(8.5, 10.1)
point_5 = POINT(5.9, 5.4)
point_6 = POINT(3.1, 9.1)
point_7 = POINT(4.5, 4.5)
point_8 = POINT(3.3, 6.5)
point_9 = POINT(5.5, 9.5)
# массив из объектов "Точка"
points = np.array([p_source, point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8, point_9])

# генерирование матрицы расстояний между всеми точками
matrix_of_distance_before_division = func_set_distance_about_all_points(points)
# матрица расстояний между всеми точками после деления на константу Q
matrix_of_distance_after_division = const_distance_Q / matrix_of_distance_before_division
#print(matrix_of_distance_after_division)

# генерирование матрицы, отображающей информацию о количестве ферромонов на каждой из граней
pheromone_matrix = func_set_pheromone_matrix(points)
print(pheromone_matrix)

# инициализация муравьёв
ants = []
for i in range(MAX_ANTS):
    ants.append(ANT())
    print("Создаём муравья номер ", i + 1, " ... ")
    for j in range(MAX_POINTS):
        ants[i].visited_points.append(0)
        ants[i].current_path.append(-1)
    ants[i].current_path[0] = ants[i].current_point
    ants[i].visited_points[ants[i].current_point] = 1
    ants[i].func_print_info()

#matrix_probability_to_move = ants[1].func_probability_to_move(matrix_of_distance_after_division, pheromone_matrix)
#next_point = ants[1].func_choice_next_point(matrix_probability_to_move)
#print("Следующая точка номер: ", next_point)

while(number_of_current_iteration < MAX_ITERATIONS):
    print("Выполняется итерация номер ", number_of_current_iteration + 1, " ... ")
    number_of_current_iteration += 1
    if (func_modeling_of_move(ants) == 0):
        print("Запускаем функцию func_update_pheromone")
        func_update_pheromone(ants)
        if(number_of_current_iteration != MAX_ITERATIONS):
            print("Запускаем функцию func_refresh - для сброса всех муравьёв")
            func_refresh(ants)
        print("Для итерации номер ", number_of_current_iteration, " лучшее расстояние равняется = ", best_length)

print("В результате выполнения алгоритма ACO было достигнуто следующее значение самого короткого расстояния = ", best_length)
print("Матрица феромонов выглядит следующим образом:", pheromone_matrix)
print("Конец выполнения программы!")
