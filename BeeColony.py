
import random
import numpy
import scipy
import matplotlib.pyplot
import math

class Floatbee:
    """
    Класс пчёл, где в качестве координат используются числа с плавающей точкой
    """
    def __init__(self):
        # Положение пчелы (искомые величины)
        self.position = None
        # Интервалы изменения искомых величин (ограничения)
        # self.min_val = None
        self.min_val = [-150] * Spherebee.count
        # self.max_val = None
        self.max_val = [150] * Spherebee.count
        # Значение целевой функции
        self.fitness = 0.0

    def calc_fitness(self):
        """
        Расчёт целевой функции. Этот метод необходимо перегрузить в производном классе. Функция не возвращает значение
        целевой функции, а только устанавливает член self.fitness. Эту функцию необходимо вызывать после каждого
        изменения координаты пчелы.
        :return:
        """
        pass

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
        :param beelist:
        :param rangelist: интервал изменения каждой из координат
        :return:
        """
        if len(bee_list) == 0:
            return True
        for curr_bee in bee_list:
            position = curr_bee.get_position()
            print("Внутри метода otherpatch len(self.position) = ", len(self.position))
            print("Внутри метода otherpatch self.position = ", self.position)
            for n in range(len(self.position)):
                print("self.position[n] = ", self.position[n])
                print("position[n] = ", position[n])
                print("range_list[n] = ", range_list[n])
                print("Результат для блока if = ", (abs(self.position[n] - position[n]) > range_list[n]))
                if abs(self.position[n] - position[n]) > range_list[n]:
                    return True
        return False

    def get_position(self):
        """
        Вернуть копию своих координат.
        :return:
        """
        return [val for val in self.position]

    def goto(self, other_pos, range_list):
        """
        Перелёт в окрестности места, которое нашла другая пчела. Не в то же самое место!
        :param other_pos:
        :param range_list:
        :return:
        """
        # К качждой из координат добавляет случайное значение
        self.position = [other_pos[n] + random.uniform(-range_list[n], range_list[n]) for n in range(len(other_pos))]
        print("self.position = ", self.position)
        # Проверим, чтобы не выйти за заданные пределы
        self.check_position()
        # Рассчитаем и сохраним целевую функцию.
        self.calc_fitness()

    def goto_random(self):
        """

        :return:
        """
        # Заполним координаты случайными значениями
        self.position = [random.uniform(self.min_val[n], self.max_val[n]) for n in range(len(self.position))]
        self.check_position()
        self.calc_fitness()

    def check_position(self):
        """
        Скорректировать координаты пчелы, если они выходят за установленные пределы.
        :return:
        """
        for n in range(len(self.position)):
            if self.position[n] < self.min_val[n]:
                self.position[n] = self.min_val[n]
            elif self.position[n] > self.max_val[n]:
                self.position[n] = self.max_val[n]

class Hive:
    """
    Класс описывающий улей. Улей управляет пчёлами.
    """
    def __init__(self, scout_bee_count, selected_bee_count, best_bee_count, sel_sites_count, best_sites_count,
                 range_list, bee_type):
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
        self.range_list = range_list[:]
        self.bee_type = bee_type

        # Лучшая на данный момент позиция
        self.best_position = None
        # Лучшее на данный момент здоровье пчелы (чем больше, тем лучше)
        self.best_fitness = -1.0e9
        # Начальное заполнение роя пчёлами со случайными координатами
        bee_count = scout_bee_count + selected_bee_count * sel_sites_count + best_bee_count * best_sites_count
        # Строка кода ниже изменена мною
        self.swarm = [Spherebee() for n in range(bee_count)]
        # Вариант строки кода ниже был в исходниках
        #self.__swarm = [bee_type() for n in range(bee_count)]

        # Лучшие и выбранные места
        self.best_sites = []
        self.sel_sites = []

        #self.__swarm.sort(self.__swarm.sort(self.__swarm), reverse=True)
        self.swarm = sorted(self.swarm, key=lambda func: func.fitness, reverse=True)
        print("Отсортированный список self.swarm")
        for swarm in self.swarm:
            print("swarm.fitness = ", swarm.fitness)
            print("swarm.position = ", swarm.position)
        self.best_position = self.swarm[0].get_position()
        print("self.best_position = ", self.best_position)
        self.best_fitness = self.swarm[0].fitness
        print("self.best_fitness = ", self.best_fitness)

    def send_bees(self, position, index, count):
        """
        Послать пчёл на позицию. Функция возвращает номер следующей пчелы для вылета.
        :return:
        """
        for n in range(count):
            # Чтобы не выйти за пределы улея.
            if index == len(self.swarm):
                break
            curr_bee = self.swarm[index]
            if curr_bee not in self.best_sites and curr_bee not in self.sel_sites:
                # Пчела не на лучших или выбранных позициях
                curr_bee.goto(position, self.range_list)
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
        print("self.swarm = ", self.swarm)
        print("self.swarm[curr_index:-1] = ", self.swarm[curr_index:-1])
        for curr_bee in self.swarm[curr_index:-1]:
            # Если пчела находится в пределах уже отмеченного лучшего участка, то её положение на считаем.
            if curr_bee.otherpatch(self.best_sites, self.range_list):
                self.best_sites.append(curr_bee)
                print("self.best_sites = ", self.best_sites)
                if len(self.best_sites) == self.best_sites_count:
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
        for curr_bee in self.swarm[bee_index:-1]:
            curr_bee.goto_random()
        # self.swarm.sort(Floatbee.sort, reverse=True)

        self.swarm = sorted(self.swarm, key=lambda func: func.fitness, reverse=True)
        print("Отсортированный список self.swarm")
        for swarm in self.swarm:
            print("swarm.fitness = ", swarm.fitness)
            print("swarm.position = ", swarm.position)

        self.best_position = self.swarm[0].get_position()
        self.best_fitness = self.swarm[0].fitness

class Statistic:
    """
    Класс для сбора статистики (информации) по запускам алгоритма.
    """
    def __init__(self):
        # Индекс каждого списка соответствует итерации.
        # В элементе каждого списка хранится список значений для каждого запуска.
        # Добавлять надо каждую итерацию.

        # Значения целевой функции в зависимости от номера итерации
        self.fitness = []
        # Значение координат в зависимости от итерации
        self.positions = []
        # Размеры областей для поиска решения в зависимости от итерации
        self.range = []

    def add(self, run_number, curr_hive):
        """

        :param run_number:
        :param curr_hive:
        :return:
        """
        range_values = [val for val in curr_hive.range_list]
        fitness = curr_hive.best_fitness
        positions = curr_hive.swarm[0].get_position()

        assert (len(self.positions) == len(self.fitness)), "Размерности значений координат и значений целевой функции должны быть одинаковыми!"
        assert (len(self.range) == len(self.fitness)), "Размерности области для поиска и значений целевой функции должны быть одинаковыми!"

        print("run_number = ", run_number, "/", "len(self.fitness) = ", len(self.fitness))
        print("self.positions = ", self.positions)
        if run_number == len(self.fitness):
            print("Выполнение блока if метода add класса Statistic")
            self.fitness.append([fitness])
            print("self.fitness = ", self.fitness)
            self.positions.append([positions])
            print("self.positions = ", self.positions)
            self.range.append([range_values])
            print("Завершение блока if метода add класса Statistic")
        else:
            print("Выполнение блока else метода add класса Statistic")
            assert (run_number == len(self.fitness) - 1), "Пока не разобрался с этой строкой кода."
            self.fitness[run_number].append(fitness)
            print("self.fitness = ", self.fitness)
            self.positions[run_number].append(positions)
            print("self.positions = ", self.positions)
            self.range[run_number].append(range_values)
            print("Завершение блока else метода add класса Statistic")

    def format_fitness(self, run_number):
        """
        Сформировать таблицу целевой функции.
        :param run_number:
        :return:
        """
        result = ""
        for n in range(len(self.fitness[run_number])):
            line = "%6.6d   %   10f\n" % (n, self.fitness[run_number][n])
            result += line
        return result

    def format_columns(self, run_number, column):
        """
        Форматировать список списков items для вывода.
        :param run_number:
        :param column:
        :return:
        """
        result = ""
        for n in range(len(column[run_number])):
            line = "%6.6d" % n
            for val in column[run_number][n]:
                line += "   %10f" % val
            line += "\n"
            result += line
        return result

    def format_pos(self, run_number):
        """

        :param run_number:
        :return:
        """
        return self.format_columns(run_number, self.positions)

    def format_range(self, run_number):
        """

        :param run_number:
        :return:
        """
        return self.format_columns(run_number, self.range)

class Spherebee(Floatbee):
    def __init__(self):
        Floatbee.__init__(self)
        self.mi_val = [-150.0] * Spherebee.count
        self.max_val = [150.0] * Spherebee.count
        self.position = [random.uniform(self.mi_val[n], self.max_val[n]) for n in range(Spherebee.count)]
        print("self.__position = ", self.position)
        self.fitness = self.calc_fitness()
        print("self.fitness = ", self.fitness)
        #self.calc_fitness()

    """
    Целевая функция - сумма квадратов по каждой координате.
    """
    # Количество координат.
    count = 4

    @staticmethod
    def get_start_range():
        """

        :return:
        """
        return [150.0] * Spherebee.count

    @staticmethod
    def get_range_koeff():
        """

        :return:
        """
        return [0.98] * Spherebee.count

    def calc_fitness(self):
        """
        Расчёт целевой функции. Этот метод необходимо перегрузить в производном классе.
        Функция не возвращает значение целевой функции, а только устанавливает член self.__fitness.
        Эту функцию необходимо вызывать после каждого изменения координаты пчелы.
        :return:
        """
        self.fitness = 0.0
        for val in self.position:
            self.fitness -= val * val
            print("self.__fitness = ", self.fitness)
        # Строка кода ниже добавлена мною.
        return self.fitness

def plot_swarm(hive_inst, x_index, y_index):
    """
    Рисуем рой при помощи matplotlib.
    :param hive_inst:
    :param x_index:
    :param y_index:
    :return:
    """
    x = []
    y = []
    x_best = []
    y_best = []
    x_sel = []
    y_sel = []
    for curr_bee in hive_inst.swarm:
        if curr_bee in hive_inst.best_sites:
            x_best.append(curr_bee.position[x_index])
            y_best.append(curr_bee.position[y_index])
        elif curr_bee in hive_inst.sel_sites:
            x_sel.append(curr_bee.position[x_index])
            y_sel.append(curr_bee.position[y_index])
        else:
            x.append(curr_bee.position[x_index])
            y.append(curr_bee.position[y_index])
    matplotlib.pyplot.clf()
    matplotlib.pyplot.scatter(x, y, c='k', s=1, marker='o')
    if len(x_sel) != 0:
        matplotlib.pyplot.scatter(x_sel, y_sel, c='y', s=20, marker='o')
    matplotlib.pyplot.scatter(x_best, y_best, c='r', s=30, marker='o')
    matplotlib.pyplot.draw()

def plot_fitness(stat, run_number):
    """
    Вывести значение целевой функции в зависимости от номера итерации.
    :param stat:
    :param run_number:
    :return:
    """
    x = range(len(stat.fitness[0]))
    y = stat.fitness[run_number]
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.grid(True)

def plot_average_fitness(stat):
    """
    Функция выводит усреднённое по всем запускам значение целевой функции в зависимости от номера итерации.
    :param stat:
    :return:
    """
    x = range(len(stat.fitness[0]))
    y = [val for val in stat.fitness[0]]
    for run_number in range(1, len(stat.fitness)):
        for iter in range(len(stat.fitness[run_number])):
            y[iter] += stat.fitness[run_number][iter]
    y = [val / len(stat.fitness) for val in y]
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.grid(True)

def plot_positions(stat, run_number, pos_index):
    """
    Функция выводит график сходимости искомых величин.
    :param stat:
    :param run_number:
    :param pos_index:
    :return:
    """
    x = range(len(stat.positions[run_number]))
    value_list = []
    for positions in stat.positions[run_number]:
        value_list.append(positions[pos_index])
    matplotlib.pyplot.plot(x, value_list)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Position %d" % pos_index)
    matplotlib.pyplot.grid(True)

def plot_range(stat, run_number, pos_index):
    """
    Функция выводит график уменьшения областей.
    :param stat:
    :param run_number:
    :param pos_index:
    :return:
    """
    x = range(len(stat.range[run_number]))
    value_list = []
    for cur_range in stat.range[run_number]:
        value_list.append(cur_range[pos_index])
    matplotlib.pyplot.plot(x, value_list)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Range %d" % pos_index)
    matplotlib.pyplot.grid(True)

def plot_stat(stat):
    """
    Функция рисует статистику.
    :param stat:
    :return:
    """
    matplotlib.pyplot.ioff()
    # Выводим изменение целевой функции в зависимости от номера итерации.
    #matplotlib.figure.Figure()
    matplotlib.figure.Figure()
    plot_fitness(stat, 0)
    # Выводим усреднённое по всем запускаем изменение целевой функции в зависимости от номера итерации.
    # matplotlib.figure.Figure()
    matplotlib.figure.Figure()
    plot_average_fitness(stat)
    # Выводим сходимость положений лучшей точки в зависимости от номера итерации.
    # matplotlib.figure.Figure()
    matplotlib.figure.Figure()
    pos_count = len(stat.positions[0][0])
    for n in range(pos_count):
        matplotlib.pyplot.subplot(pos_count, 1, n+1)
        plot_positions(stat, 0, n)
    # Выводим изменение размеров областей в зависимости от номера итерации.
    #matplotlib.figure.Figure()
    matplotlib.figure.Figure()
    range_count = len(stat.range[0][0])
    for n in range(range_count):
        matplotlib.pyplot.subplot(range_count, 1, n+1)
        plot_range(stat, 0, n)
    matplotlib.pyplot.show()

def run_ABC():
    """
    Функция для запуска алгоритма.
    :return:
    """
    try:
        import psyco
        psyco.full()
    except:
        print("psyco not found!")

    # Включаем интерактивный режим.
    matplotlib.pyplot.ion()
    # Переменная для сохранения статистики.
    stat = Statistic()
    # Имя файла для сохранения статистики.
    stat_fname = "stat/beestat_%s.txt"

    #################################################################################
    #################################################################################
    #################################################################################
    #########################   ПАРАМЕТРЫ АЛГОРИТМА    ##############################
    #################################################################################
    #################################################################################
    #################################################################################

    # Класс пчёл, который будет использоваться в алгоритме.
    bee_type = Spherebee()
    # Количество пчёл-разведчиков
    #scout_bee_count = 300
    scout_bee_count = 100
    #scout_bee_count = 2
    # Количество пчёл, отправляемых на выбранные, но не лучшие участки
    #selected_bee_count = 10
    selected_bee_count = 20
    #selected_bee_count = 2
    # Количество пчёл, отправляемых на лучшие участки
    #best_bee_count = 30
    best_bee_count = 20
    #best_bee_count = 2
    # Количество выбранных, но не лучших участков
    #sel_sites_count = 15
    sel_sites_count = 10
    #sel_sites_count = 2
    # Количество лучших участков
    best_sites_count = 5
    #best_sites_count = 2
    # Количество запусков алгоритма
    run_count = 1
    # Максимальное количество итераций
    #max_iteration = 2000
    max_iteration = 100
    #max_iteration = 2
    # Через такое количество итераций без нахождения лучшего решения уменьшим область поиска
    #max_func_counter = 10
    max_func_counter = 3
    #max_func_counter = 1
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
                        bee_type.get_start_range(), bee_type)
        # Начальное значение целевой функции
        best_func = -1.0e9
        # Количество итераций без улучшения целевой функции
        func_counter = 0
        stat.add(run_number, curr_hive)
        for n in range(max_iteration):
            curr_hive.next_step()
            stat.add(run_number, curr_hive)
            if curr_hive.best_fitness != best_func:
                # Найдено место, где целевая функция лучше.
                best_func = curr_hive.best_fitness
                func_counter = 0
                # Обновновляем рисунок роя пчёл
                plot_swarm(curr_hive, 0, 1)
                print("\n*** Iteration %d  / %d" % (run_number+1, n))
                print("Best Position: %s" % curr_hive.best_position)
                print("Best fitness: %f" % curr_hive.best_fitness)
            else:
                func_counter += 1
                if func_counter == max_func_counter:
                    # Уменьшаем размер участков
                    curr_hive.__range = [curr_hive.range_list[m] * koeff[m] for m in range(len(curr_hive.range_list))]
                    func_counter = 0
                    print("\n*** Iteration %d / %d (new range" % (run_number+1, n))
                    print("New range: %s" % curr_hive.range_list)
                    print("Best Position: %s" % curr_hive.best_position)
                    print("Best Fitness: %f" % curr_hive.best_fitness)
            #if n % 10 == 0:
                #plot_swarm(curr_hive, 2, 3)
        # Сохраняем значения целевой функции
        fname = stat_fname % (("%4.4d" % run_number) + "_fitness")
        print("stat_fname = ", stat_fname)
        print("fname = ", fname)
        """
        fp = file(fname, "w")
        fp.write(stat.format_fitness(run_number))
        fp.close()
        """
        with open(fname, "w") as fp:
            fp.write(stat.format_fitness(run_number))
        # Сохраняем значения координат
        fname = stat_fname % (("%4.4d" % run_number) + "_pos")
        """
        fp = file(fname, "w")
        fp.write(stat.format_pos(run_number))
        fp.close()
        """
        with open(fname, "w") as fp:
            fp.write(stat.format_pos(run_number))
        # Сохраняем значения интервалов
        fname = stat_fname % (("%4.4d" % run_number) + "_range")
        """
        fp = file(fname, "w'")
        fp.write(stat.format_range(run_number))
        fp.close()
        """
        with open(fname, "w") as fp:
            fp.write(stat.format_range(run_number))
    #plot_stat(stat)

"""
count_bee_scout = 10
count_bee_elite = 5
count_bee = 2
count_best_area = 2
count_target_area = 3
radius_area = 10
"""

str = "World"
arr = [0,6,8,9,2,4,6,81,6,7878,6,25,6,9,4,23,89,123,564,879]
print(sorted(arr, reverse=True))
run_ABC()

class GFG:
    def __init__(self, a, b):
        self.a = a * random.randint(a, 100)
        print("self.a = ", self.a)
        self.b = b * random.randint(b, 100)
        #print("self.b = ", self.b)

class Child(GFG):
    def __init__(self, x=2, y=3):
        GFG.__init__(self, x, y)
        self.__name = x
        #print("self.__name = ", self.__name)
        self.__age = y
        #print("self.__age = ", self.__age)

class Grand:
    def __init__(self):
        self.__swarm = [Child() for n in range(5)]
        self.__swarm = sorted(self.__swarm, key=lambda func: func.a, reverse=True)
        for s in self.__swarm:
            print(s.a)


# sorting objects on the basis of value
# stored at variable b
grand = Grand()



print("Hello, ", str, "!")