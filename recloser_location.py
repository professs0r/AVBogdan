import networkx as nx

import test_node_potential_algo as algo
import numpy as np
import spannin_tree as st
import copy

#                                        START SOURCE DATA
#                                        START SOURCE DATA
#                                        START SOURCE DATA

COUNT_NODES = 17

# edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated), PS='E1' (source energy))

edge_0 = (0, 1, 0.1, 630, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_1 = (1, 2, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_2 = (1, 5, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_3 = (2, 3, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_4 = (2, 6, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_5 = (3, 4, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_6 = (3, 7, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_7 = (4, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_8 = (5, 6, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_9 = (5, 9, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'Chord')
edge_10 = (6, 7, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_11 = (6, 10, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'Chord')
edge_12 = (7, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_13 = (7, 11, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'Chord')
edge_14 = (8, 12, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'Chord')
edge_15 = (9, 10, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_16 = (9, 13, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_17 = (10, 11, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_18 = (10, 14, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_19 = (11, 12, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_20 = (11, 15, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_21 = (12, 16, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_22 = (13, 14, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_23 = (14, 15, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_24 = (15, 16, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_25 = (15, 0, 0.1, 630, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')

edge_nagr_2 = (2, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_3 = (3, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_4 = (4, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_5 = (5, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_6 = (6, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_7 = (7, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_8 = (8, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_nagr_9 = (9, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_10 = (10, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_11 = (11, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_12 = (12, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_13 = (13, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_14 = (14, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_15 = (16, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')

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
                        edge_24,
                        edge_25])
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
                       edge_nagr_15])
"""
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
                  edge_25,
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

#graph = algo.func_edges_to_undirected_graph(edges, COUNT_NODES)

def get_key(dict, need_value):
    for key, value_of_dict in dict.items():
        if value_of_dict == need_value:
            return int(key)

def updated_value_in_tuple(tuple, index, value):
    return tuple[:index] + (value) + tuple[index + 1:]

def func_choose_recloser_location(edges_lines, edges_nagr):
    """

    :param edges_lines:
    :param edges_nagr:
    :return:
    """
    sources = []
    for nagr in edges_nagr:
        sources.append(nagr[15])    # 15 - порядковый номер аттрибута ребра, в котором записывается информация об источнике питания
    temp_set = set(sources)
    list_sources = list(temp_set)
    best_trees = []
    for source in range(len(list_sources)):
        graph = nx.Graph()
        temp_edges = edges_lines.copy()
        temp_loads = edges_nagr.copy()
        temp_branches = []
        temp_nagr = []
        set_nodes = set()
        for edge in edges_lines:
            if edge[15] == list_sources[source]:
                set_nodes.add(int(edge[0]))
                set_nodes.add(int(edge[1]))
        dict_nodes = dict.fromkeys(set_nodes, 0)
        for key, value in zip(dict_nodes, range(len(dict_nodes))):
            dict_nodes[key] = value
        for index in range(len(set_nodes)):
            graph.add_node(index, potential=0.0, active=15.0, I=0.0, root=None, parent=None, visited=False, weight=index + 1)
        for iter in range(len(edges_lines)):
            if edges_lines[iter][15] == list_sources[source]:
                graph.add_edge(int(dict_nodes.get(int(temp_edges[iter][0]))), int(dict_nodes.get(int(temp_edges[iter][1]))),
                               resistance=float(temp_edges[iter][2]),
                               voltage=float(temp_edges[iter][3]), type=int(temp_edges[iter][4]),
                               length=float(temp_edges[iter][5]),
                               cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]),
                               material=temp_edges[iter][8],
                               r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]),
                               cos_y=float(temp_edges[iter][11]),
                               sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                               lose_energy=float(temp_edges[iter][14]), PS=str(temp_edges[iter][15]))
                temp_edges[iter][0] = int(dict_nodes.get(int(temp_edges[iter][0])))
                temp_edges[iter][1] = int(dict_nodes.get(int(temp_edges[iter][1])))
                temp_branches.append(temp_edges[iter])
        for nagr in range(len(edges_nagr)):
            if edges_nagr[nagr][15] == list_sources[source]:
                temp_loads[nagr][0] = int(dict_nodes.get(int(temp_loads[nagr][0])))
                temp_loads[nagr][1] = int(dict_nodes.get(int(temp_loads[nagr][1])))
                temp_nagr.append(temp_loads[nagr])
        trees = st.func_networkx_build_spanning_tree(graph)
        #print(len(trees))
        list_edges_in_spanning_tree = []
        lowest_loses = 10000000000000000000000000000000000000000000000000000
        for tree in trees:
            for edge in tree.edges.items():
                # добавление ребра в список
                list_edges_in_spanning_tree.append(edge[0])
            # готовый список рёбер остовного дерева
            # print(list_edges_in_spanning_tree)
            #print(list_edges_in_spanning_tree)
            new_graph = algo.func_list_of_edges_to_graph_recloser(list_edges_in_spanning_tree, len(set_nodes), temp_branches, temp_nagr)
            algo.func_calculated_current_node_potential_algo(new_graph)
            """
            for branch in new_graph.edges():
                print(branch, new_graph.edges[branch]['I'])
            """
            """
            print("Значения напряжений у конечных потребителей мощности:")
            for branch in new_graph.edges():
                if int(branch[0]) == 0 and new_graph.edges[branch]['voltage'] == 0:
                    print("Напряжение между узлами ", branch, " = ",
                          abs(new_graph.edges[branch]['I'] * new_graph.edges[branch]['resistance']))
            """
            flag_lose_voltage = 1
            for branch in new_graph.edges():
                if int(branch[0]) == 0 and\
                        new_graph.edges[branch]['voltage'] == 0 and\
                        abs(new_graph.edges[branch]['I']*new_graph.edges[branch]['resistance']) < 570:
                    flag_lose_voltage -= 1
                    break
            """
            # Оптимизации только по потерям мощности
            loses = algo.func_loses_energy_400(new_graph)
            if loses < lowest_loses:
                lowest_loses = loses
                right_list_edges_in_spanning_tree = []
                for edge in list_edges_in_spanning_tree:
                    # print("Отладка!")
                    list_edge = list(edge)
                    list_edge[0] = get_key(dict_nodes, int(edge[0]))
                    list_edge[1] = get_key(dict_nodes, int(edge[1]))
                    tuple_edge = tuple(list_edge)
                    right_list_edges_in_spanning_tree.append(tuple_edge)
                    # print(right_list_edges_in_spanning_tree)
                print("Для дерева:", right_list_edges_in_spanning_tree, "Потери мощности = ", lowest_loses)
            """

            if flag_lose_voltage:
                # Оптимизация по потерям напряжения и по потерям мощности
                loses = algo.func_loses_energy_400(new_graph)
                if loses < lowest_loses:
                    lowest_loses = loses
                    right_list_edges_in_spanning_tree = []
                    for edge in list_edges_in_spanning_tree:
                        list_edge = list(edge)
                        list_edge[0] = get_key(dict_nodes, int(edge[0]))
                        list_edge[1] = get_key(dict_nodes, int(edge[1]))
                        tuple_edge = tuple(list_edge)
                        right_list_edges_in_spanning_tree.append(tuple_edge)
                    print("Для дерева:", right_list_edges_in_spanning_tree, "Потери мощности = ", lowest_loses)
                    print("Максимальное значение потерь напряжения в процентах = ",
                          abs(abs((new_graph.nodes[0]['potential'] / 630) * 100) - 100))

            new_graph.clear()
            list_edges_in_spanning_tree.clear()
        graph.clear()
        best_trees.append(right_list_edges_in_spanning_tree.copy())
        right_list_edges_in_spanning_tree.clear()
        print("Для ",  source+1 , " источника все остовные дервья построены и посчитаны!")
    #print("Оптимизация по потерям напряжения и по потерям мощности")
    #print("[[(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 8), (5, 6), (6, 7)], [(0, 15), (9, 13), (10, 11), (11, 15), (12, 16), (13, 14), (14, 15), (15, 16)]]")
    #print("Оптимизации только по потерям мощности")
    #print("[[(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 8), (5, 6), (6, 7)], [(0, 15), (9, 13), (10, 11), (11, 15), (12, 16), (13, 14), (14, 15), (15, 16)]]")
    #print(best_trees)
    for source_fault in range(len(best_trees)):
        #print("Отладка!")
        temp_best_trees = copy.deepcopy(best_trees)
        temp_best_trees[source_fault].pop(0)
        #branches = copy.deepcopy(edges_lines)
        #lines = list(branches)
        edges_need_to_build = []
        for tree in temp_best_trees:
            for edge in tree:
                edges_need_to_build.append(edge)
        """
        # сначала сделал, потом понял, что лишнее (удалять пока рука не поднимается)
        edges_need_to_build = []
        for iter in range(len(best_trees)):
            for edge in temp_best_trees[iter]:
                index = 0
                for line in lines:
                    if int(edge[0]) == int(line[0]) and int(edge[1]) == int(line[1]) or \
                            int(edge[1]) == int(line[0]) and int(edge[0]) == int(line[1]):
                        edges_need_to_build.append(line)
                        lines.pop(index)
                        break
                    index += 1
        edges_need_to_build.clear()
        """
        for chord in edges_lines:
            if chord[15] == 'Chord':
                temp_list = [int(chord[0]), int(chord[1])]
                temp_tuple = tuple(temp_list)
                edges_need_to_build.append(temp_tuple)
                new_graph = algo.func_list_of_edges_to_graph(edges_need_to_build, COUNT_NODES, edges_lines, edges_nagr)
                algo.func_calculated_current_node_potential_algo(new_graph)
                loses = algo.func_loses_energy_400(new_graph)
                print("Для дерева = ", edges_need_to_build)
                print("Потери мощности = ", loses)
                print("Токи в ветвях:")
                for branch in new_graph.edges():
                    print("Ветвь ", branch, " = ", new_graph.edges[branch]['I'])
                print("Падение напяржений у конечных потребителей:")
                for branch in new_graph.edges():
                    if int(branch[0]) == 0 and new_graph.edges[branch]['voltage'] == 0:
                        print("Ветвь ", branch, " = ", abs(new_graph.edges[branch]['I'] * new_graph.edges[branch]['resistance']))
                edges_need_to_build.pop()

        temp_best_trees.clear()


func_choose_recloser_location(edges_lines, edges_nagr)