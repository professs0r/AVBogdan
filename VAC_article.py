import networkx as nx

import test_node_potential_algo as algo
import numpy as np
import spannin_tree as st

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
edge_9 = (5, 9, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10')
edge_10 = (6, 7, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_11 = (6, 10, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10')
edge_12 = (7, 8, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1')
edge_13 = (7, 11, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10')
edge_14 = (8, 12, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E-10')
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
edge_nagr_15 = (15, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')
edge_nagr_16 = (16, 0, 200.0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E2')

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

graph = algo.func_edges_to_undirected_graph(edges, COUNT_NODES)

# подсчёт и перечисление всех остовных деревьев
trees = st.func_networkx_build_spanning_tree(graph)
print(len(trees))
list_edges_in_spanning_tree = []
lowest_loses = 10000000000000000000000000000000000000000000000000000
for tree in trees:
    for edge in tree.edges.items():
        # добавление ребра в список
        list_edges_in_spanning_tree.append(edge[0])
    # готовый список рёбер остовного дерева
    #print(list_edges_in_spanning_tree)
    new_graph = algo.func_list_of_edges_to_graph(list_edges_in_spanning_tree, COUNT_NODES, edges_lines, edges_nagr)
    algo.func_calculated_current_node_potential_algo(new_graph)
    #for branch in new_graph.edges():
        #print(branch, new_graph.edges[branch]['I'])
    loses = algo.func_loses_energy_400(new_graph)
    if loses < lowest_loses:
        lowest_loses = loses
        print("for tree:", list_edges_in_spanning_tree, "loses = ", lowest_loses)
        print("Максимальное значение потерь напряжения (абсолютная величина) = ", abs(new_graph.nodes[0]['potential']))
        print("Максимальное значение потерь напряжения в процентах = ", abs(abs((new_graph.nodes[0]['potential']/630)*100)-100))
    new_graph.clear()
    list_edges_in_spanning_tree.clear()

"""
# расчёт вспомогательных величин и значений токов в ветвях
#algo.func_calculating_support_variables(graph)
algo.func_calculated_current_node_potential_algo(graph)
print("Величины токов в исходном графе: ")
for branch in graph.edges():
    print(branch)
    print(graph.edges[branch]['I'])
loses = algo.func_loses_energy_400(graph)
print("Потери в ветвях для исходной схемы = ", loses)
print("Максимальное значение потерь напряжения (абсолютная величина) = ", abs(graph.nodes[0]['potential']))
print("Максимальное значение потерь напряжения в процентах = ", abs(abs((graph.nodes[0]['potential']/630)*100)-100))
"""
