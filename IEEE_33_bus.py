import networkx as nx
import numpy as np
import spanning_tree as st
import test_node_potential_algo as algo

"""
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    DATA    
"""



#ТРЕНИРУЕМСЯ НА КОШКАХ

#  КОММЕНТИРУЕМ
COUNT_NODES = 33
# edge_0 = (source=0, finish=1, resistance=70.1,
# voltage=630, type='СИП', length=1000, cross_section=35, I=0,
#          material=Al, r_0=0 (calculated), x_0=0 (calculated), cos_y=0.89, sin_y=0 (calculated),
#          lose_volt=0 (calculated), lose_energy=0 (calculated), PS='E1' (source energy),
#          type_edge='Source or Branch or Load or Chord',
#          power=140e3 [for load in W, for source in WA])

edge_line_0_1 = (0, 1, 0.09, 12.66e3, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Source', 100e3)
edge_line_1_2 = (1, 2, 0.49, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_2_3 = (2, 3, 0.36, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_3_4 = (3, 4, 0.38, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_4_5 = (4, 5, 0.81, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_5_6 = (5, 6, 0.01, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_6_7 = (6, 7, 0.71, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_7_8 = (7, 8, 1.03, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_8_9 = (8, 9, 1.04, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_9_10 = (9, 10, 0.19, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_10_11 = (10, 11, 0.37, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_11_12 = (11, 12, 1.46, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_12_13 = (12, 13, 0.54, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_13_14 = (13, 14, 0.59, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_14_15 = (14, 15, 0.74, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_15_16 = (15, 16, 1.28, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_16_17 = (16, 17, 0.73, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_1_18 = (1, 18, 0.16, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_18_19 = (18, 19, 1.50, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_19_20 = (19, 20, 0.40, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_20_21 = (20, 21, 0.70, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_2_22 = (2, 22, 0.04, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_22_23 = (22, 23, 0.89, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_23_24 = (23, 24, 0.89, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_5_25 = (5, 25, 0.20, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_25_26 = (25, 26, 0.20, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_26_27 = (26, 27, 1.05, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_27_28 = (27, 28, 0.80, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_28_29 = (28, 29, 0.50, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_29_30 = (29, 30, 0.97, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_30_31 = (30, 31, 0.31, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_31_32 = (31, 32, 0.34, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_7_20 = (7, 20, 2.0, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_8_14 = (8, 14, 2.0, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_11_21 = (11, 21, 2.0, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_17_32 = (17, 32, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)
edge_line_24_28 = (24, 28, 0.5, 0, 0, 1000, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Branch', 0)

edge_nagr_2 = (2, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_3 = (3, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 120e3)
edge_nagr_4 = (4, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_5 = (5, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_6 = (6, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 200e3)
edge_nagr_7 = (7, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 200e3)
edge_nagr_8 = (8, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_9 = (9, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_10 = (10, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 45e3)
edge_nagr_11 = (11, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_12 = (12, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_13 = (13, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 120e3)
edge_nagr_14 = (14, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_15 = (15, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_16 = (16, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_17 = (17, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_18 = (18, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_19 = (19, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_20 = (20, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_21 = (21, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_22 = (22, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 90e3)
edge_nagr_23 = (23, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 420e3)
edge_nagr_24 = (24, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 420e3)
edge_nagr_25 = (25, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_26 = (26, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_27 = (27, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)
edge_nagr_28 = (28, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 120e3)
edge_nagr_29 = (29, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 200e3)
edge_nagr_30 = (30, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 150e3)
edge_nagr_31 = (31, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 210e3)
edge_nagr_32 = (32, 0, 0, 0, 0, -1000, -35, 0, 'Al', 0, 0, 0.89, 0, 0, 0, 'E1', 'Load', 60e3)


edges_lines = np.array([edge_line_0_1,
                        edge_line_1_2,
                        edge_line_2_3,
                        edge_line_3_4,
                        edge_line_4_5,
                        edge_line_5_6,
                        edge_line_6_7,
                        edge_line_7_8,
                        edge_line_8_9,
                        edge_line_9_10,
                        edge_line_10_11,
                        edge_line_11_12,
                        edge_line_12_13,
                        edge_line_13_14,
                        edge_line_14_15,
                        edge_line_15_16,
                        edge_line_16_17,
                        edge_line_1_18,
                        edge_line_18_19,
                        edge_line_19_20,
                        edge_line_20_21,
                        edge_line_2_22,
                        edge_line_22_23,
                        edge_line_23_24,
                        edge_line_5_25,
                        edge_line_25_26,
                        edge_line_26_27,
                        edge_line_27_28,
                        edge_line_28_29,
                        edge_line_29_30,
                        edge_line_30_31,
                        edge_line_31_32,
                        edge_line_7_20,
                        edge_line_8_14,
                        edge_line_11_21,
                        edge_line_17_32,
                        edge_line_24_28
                        ])
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
                       edge_nagr_16,
                       edge_nagr_17,
                       edge_nagr_18,
                       edge_nagr_19,
                       edge_nagr_20,
                       edge_nagr_21,
                       edge_nagr_22,
                       edge_nagr_23,
                       edge_nagr_24,
                       edge_nagr_25,
                       edge_nagr_26,
                       edge_nagr_27,
                       edge_nagr_28,
                       edge_nagr_29,
                       edge_nagr_30,
                       edge_nagr_31,
                       edge_nagr_32])

# для расчёта токов
edges = np.array([edge_line_0_1,
                  edge_line_1_2,
                  edge_line_2_3,
                  edge_line_3_4,
                  edge_line_4_5,
                  edge_line_5_6,
                  edge_line_6_7,
                  edge_line_7_8,
                  edge_line_8_9,
                  edge_line_9_10,
                  edge_line_10_11,
                  edge_line_11_12,
                  edge_line_12_13,
                  edge_line_13_14,
                  edge_line_14_15,
                  edge_line_15_16,
                  edge_line_16_17,
                  edge_line_1_18,
                  edge_line_18_19,
                  edge_line_19_20,
                  edge_line_20_21,
                  edge_line_2_22,
                  edge_line_22_23,
                  edge_line_23_24,
                  edge_line_5_25,
                  edge_line_25_26,
                  edge_line_26_27,
                  edge_line_27_28,
                  edge_line_28_29,
                  edge_line_29_30,
                  edge_line_30_31,
                  edge_line_31_32,
                  edge_line_7_20,
                  edge_line_8_14,
                  edge_line_11_21,
                  edge_line_17_32,
                  edge_line_24_28,
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
                  edge_nagr_16,
                  edge_nagr_17,
                  edge_nagr_18,
                  edge_nagr_19,
                  edge_nagr_20,
                  edge_nagr_21,
                  edge_nagr_22,
                  edge_nagr_23,
                  edge_nagr_24,
                  edge_nagr_25,
                  edge_nagr_26,
                  edge_nagr_27,
                  edge_nagr_28,
                  edge_nagr_29,
                  edge_nagr_30,
                  edge_nagr_31,
                  edge_nagr_32])

#  КОММЕНТИРУЕМ

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
"""

def func_edges_to_undirected_graph_alt(edges, count_nodes):
    """
    функция для инициализации неориентированного графа на основе списка рёбер
    :param edges:
    :param count_nodes:
    :return:
    """
    graph = nx.Graph()
    """
    # active load in each node 15 kW
    for index in range(count_nodes):
        graph.add_node(index, potential=0.0, active=15.0, I=0.0, root=None, parent=None, visited=False, weight=index+1)
    """
    for index in range(count_nodes):
        graph.add_node(index, potential=0.0, active=15.0, I=0.0, root=None, parent=None, visited=False, weight=index+1)
    temp_edges = edges.copy()
    for iter in range(len(edges)):
        graph.add_edge(int(temp_edges[iter][0]), int(temp_edges[iter][1]), resistance=float(temp_edges[iter][2]),
                       voltage=float(temp_edges[iter][3]), type=int(temp_edges[iter][4]),
                       length=float(temp_edges[iter][5]),
                       cross_section=float(temp_edges[iter][6]), I=float(temp_edges[iter][7]),
                       material=temp_edges[iter][8],
                       r_0=float(temp_edges[iter][9]), x_0=float(temp_edges[iter][10]),
                       cos_y=float(temp_edges[iter][11]),
                       sin_y=float(temp_edges[iter][12]), lose_volt=float(temp_edges[iter][13]),
                       lose_energy=float(temp_edges[iter][14]), PS=str(temp_edges[iter][15]),
                       type_edge=str(temp_edges[iter][16]),
                       power=float(temp_edges[iter][17]))
    return graph

def func_calculated_current_node_potential_algo_alt(graph, flag=False):
    """
    method node potential
    :return:
    """
    if flag != False:
        voltage_source = 0.0
        for branch in graph.edges():
            if str(graph.edges[branch]['type_edge']) == str('Source'):
                voltage_source += graph.edges[branch]['voltage']
        for edge in graph.edges():
            if str(graph.edges[edge]['type_edge']) == str('Load'):
                #graph.edges[edge]['resistance'] += pow(voltage_source*0.9, 2) / graph.edges[edge]['power']
                graph.edges[edge]['resistance'] += pow(voltage_source, 2) / graph.edges[edge]['power']
    temp_voltage = 0.0
    for branch in graph.edges():
        if temp_voltage > 0.1:
            break
        elif graph.edges[branch]['voltage'] > 0.1:
            temp_voltage = graph.edges[branch]['voltage'] / 1
    for branch in graph.edges():
        if graph.edges[branch]['type_edge'] != 'Source' and graph.edges[branch]['power'] > 0.01:
            graph.edges[branch]['resistance'] = pow(temp_voltage, 2) / graph.edges[branch]['power']
    del temp_voltage
    #print("Количество узлов = ", graph.number_of_nodes())
    count_nodes = int(graph.number_of_nodes())
    count_branches = int(graph.number_of_edges())
    zero_potential = int(count_nodes - 1)
    conductivity_matrix = np.zeros((count_nodes - 1, count_nodes - 1))
    current_matrix = np.zeros((count_nodes - 1, 1))
    export_array = []
    import_array = []
    matrix_incidence = algo.func_make_matrix_incidence(graph)
    list_edges = []
    for edge in graph.edges.items():
        list_edges.append(edge[0])
    for potential in range(count_nodes):  # за данный проход формируется уравнение узловых потенциалов относительно рассматриваемого узла
        if (potential == zero_potential):
            continue
        for edge in range(count_branches):
            if matrix_incidence[potential][edge] == 1:
                export_array.append(list_edges[edge][1])
            elif matrix_incidence[potential][edge] == -1:
                import_array.append(list_edges[edge][0])
        for index_matrix in range(len(conductivity_matrix[potential])):
            if (index_matrix == zero_potential):
                continue
            if (potential == index_matrix):
                if (len(export_array) == 1):
                    conductivity_matrix[potential][index_matrix] += 1 / (
                    graph[potential][export_array[0]]['resistance'])
                elif (len(export_array) > 1):
                    for exp_arr in range(len(export_array)):
                        conductivity_matrix[potential][index_matrix] += 1 / (graph[potential][export_array[exp_arr]]['resistance'])
                if (len(import_array) == 1):
                    conductivity_matrix[potential][index_matrix] += 1 / (
                    graph[import_array[0]][potential]['resistance'])
                elif (len(import_array) > 1):
                    for imp_arr in range(len(import_array)):
                        conductivity_matrix[potential][index_matrix] += 1 / (
                        graph[import_array[imp_arr]][potential]['resistance'])
        if (len(export_array) == 1):
            if (export_array[0] != zero_potential):
                conductivity_matrix[potential][export_array[0]] -= 1 / (graph[potential][export_array[0]]['resistance'])
                current_matrix[potential] -= (graph[potential][export_array[0]]['voltage']) / (
                graph[potential][export_array[0]]['resistance'])
        elif (len(export_array) > 1):
            for exp_arr in range(len(export_array)):
                if (export_array[exp_arr] == zero_potential):
                    continue
                conductivity_matrix[potential][export_array[exp_arr]] -= 1 / (
                graph[potential][export_array[exp_arr]]['resistance'])
                current_matrix[potential] -= (graph[potential][export_array[exp_arr]]['voltage']) / (
                graph[potential][export_array[exp_arr]]['resistance'])
        if (len(import_array) == 1):
            if (import_array[0] != zero_potential):
                conductivity_matrix[potential][import_array[0]] -= 1 / (graph[import_array[0]][potential]['resistance'])
                current_matrix[potential] += (graph[import_array[0]][potential]['voltage']) / (
                graph[import_array[0]][potential]['resistance'])
        elif (len(import_array) > 1):
            for imp_arr in range(len(import_array)):
                if (import_array[imp_arr] == zero_potential):
                    continue
                conductivity_matrix[potential][import_array[imp_arr]] -= 1 / (
                graph[import_array[imp_arr]][potential]['resistance'])
                current_matrix[potential] += (graph[import_array[imp_arr]][potential]['voltage']) / (
                graph[import_array[imp_arr]][potential]['resistance'])

        export_array.clear()
        import_array.clear()

    print("conductivity_matrix = ", conductivity_matrix)
    print("current_matrix = ", current_matrix)
    potential_matrix = np.linalg.solve(conductivity_matrix, current_matrix)

    for nodes in range(len(potential_matrix)):
        graph.nodes[nodes]['potential'] = float(potential_matrix[nodes])
        #print("Узел - ", nodes, ", потенциал ", graph.nodes[nodes]['potential'])

    for branch in graph.edges():

        graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] - graph.nodes[branch[1]]['potential'] +
                                            graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]][
                                               'resistance']

        """
        graph[branch[0]][branch[1]]['I'] = (graph.nodes[branch[0]]['potential'] + graph[branch[0]][branch[1]]['voltage']) / graph[branch[0]][branch[1]][
                                               'resistance']
        """
        """
        print("Потанцевал 0 = ", graph.nodes[branch[0]]['potential'])
        print("Потанцевал 1 = ", graph.nodes[branch[1]]['potential'])
        print("Напряжение = ", graph[branch[0]][branch[1]]['voltage'])
        print("Ток = ", graph[branch[0]][branch[1]]['I'])
        """
    """
    print("Значения напряжений у конечных потребителей мощности:")
    for branch in graph.edges():
        if int(branch[0]) == 0 and graph.edges[branch]['voltage'] == 0:
            print("Напряжение между узлами ", branch, " = ", abs(graph.edges[branch]['I'] * graph.edges[branch]['resistance']))
    """


graph = func_edges_to_undirected_graph_alt(edges, COUNT_NODES)
"""
for branch in graph.edges():
    print(graph.edges[branch])
"""

func_calculated_current_node_potential_algo_alt(graph, True)

for branch in graph.edges():
    print("Ток в ребре [", branch[0], branch[1], "] = ", graph.edges[branch]['I'])
    print("Сопротивление ребра [", branch[0], branch[1], "] = ", graph.edges[branch]['resistance'])


"""
# подсчёт и перечисление всех остовных деревьев
trees = st.func_networkx_build_spanning_tree(graph)
list_edges_in_spanning_tree = []
lowest_loses = 10000000000000000000000000000000000000000000000000000
for tree in trees:
    for edge in tree.edges.items():
        list_edges_in_spanning_tree.append(edge[0])
    new_graph = algo.func_list_of_edges_to_graph(list_edges_in_spanning_tree, COUNT_NODES, edges_lines, edges_nagr)
    algo.func_calculated_current_node_potential_algo(new_graph)
    #for branch in new_graph.edges():
        #print(branch, new_graph.edges[branch]['I'])
    loses = 0.0
    for edge in new_graph.edges():
        if str(new_graph.edges[edge]['type_edge']) == str('Branch'):
            temp_variable = new_graph.edges[edge]['resistance'] * abs(pow(new_graph.edges[edge]['I'], 2))
            loses += temp_variable
    if loses < lowest_loses:
        lowest_loses = loses
        print("for tree:", list_edges_in_spanning_tree, "loses = ", lowest_loses)
    new_graph.clear()
    list_edges_in_spanning_tree.clear()
print("Проверка!")
"""




print("Hello, World!")