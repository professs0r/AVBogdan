import test_node_potential_algo as algo
import numpy as np
import spannin_tree as st
#import myPSO as pso


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

#edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
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
graph = algo.func_edges_to_undirected_graph(edges, nodes)
trees = st.func_networkx_build_spanning_tree(graph)
print(len(trees))
for t in trees:
    print(t.edges())
"""
graph = algo.func_edges_to_directed_graph(edges, COUNT_NODES)
algo.func_calculating_support_variables(graph)
algo.func_calculated_current_node_potential_algo(graph, nodes, nodes-1, matrix, directed_adjacency_list)
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
pso.func_run_algo(graph, count_iter, size, k, c1, c2, value_min_limit, value_max_limit)
"""


#                                        END WORKING ALGO
#                                        END WORKING ALGO
#                                        END WORKING ALGO

print("\n\nЕсли Ты видишь это сообщение, значит программа отработала корректно!\n\n")