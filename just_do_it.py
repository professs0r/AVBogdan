import networkx as nx

import test_node_potential_algo as algo
import numpy as np
import spannin_tree as st
#import myPSO as pso


#                                        START SOURCE DATA
#                                        START SOURCE DATA
#                                        START SOURCE DATA

COUNT_NODES = 10

#edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated))

edge_0 = (0, 1, 0.1, 630, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_1 = (1, 2, 1.17, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_2 = (1, 4, 1.35, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_3 = (2, 3, 2.55, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_4 = (2, 5, 2.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_5 = (3, 6, 1.68, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_6 = (4, 5, 1.25, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_7 = (4, 7, 1.08, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_8 = (5, 6, 2.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_9 = (5, 8, 1.25, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_10 = (6, 9, 2.44, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_11 = (7, 8, 1.79, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_12 = (8, 9, 2.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)

edge_nagr_1 = (2, 0, 26.46, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_2 = (3, 0, 172.56, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_3 = (4, 0, 79.38, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_4 = (5, 0, 99.22, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_5 = (6, 0, 158.76, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_6 = (7, 0, 99.22, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_7 = (8, 0, 396.9, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
edge_nagr_8 = (9, 0, 110.25, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
                  
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
                        edge_12])
edges_nagr = np.array([edge_nagr_1,
                       edge_nagr_2,
                       edge_nagr_3,
                       edge_nagr_4,
                       edge_nagr_5,
                       edge_nagr_6,
                       edge_nagr_7,
                       edge_nagr_8])
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
                  edge_nagr_1,
                  edge_nagr_2,
                  edge_nagr_3,
                  edge_nagr_4,
                  edge_nagr_5,
                  edge_nagr_6,
                  edge_nagr_7,
                  edge_nagr_8])
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
    new_graph.clear()
    list_edges_in_spanning_tree.clear()

"""
# расчёт вспомогательных величин и значений токов в ветвях
#algo.func_calculating_support_variables(graph)
#algo.func_calculated_current_node_potential_algo(graph)
algo.func_calculated_current_node_potential_algo(graph)
print("Величины токов в исходном графе: ")
for branch in graph.edges():
    print(branch)
    print(graph.edges[branch]['I'])
loses = algo.func_loses_energy_400(graph)
print("Потери в ветвях для исходной схемы = ", loses)
"""
"""
count_iter = 1
size = 3
k = 0.1
c1 = 2
c2 = 3
value_min_limit = -500
value_max_limit = 500
pso.func_run_algo(graph, count_iter, size, k, c1, c2, value_min_limit, value_max_limit)
"""


#                                        END WORKING ALGO
#                                        END WORKING ALGO
#                                        END WORKING ALGO

print("\n\nЕсли Ты видишь это сообщение, значит программа отработала корректно!\n\n")