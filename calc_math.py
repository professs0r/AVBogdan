# math culculation optimization algorithm

import numpy
from numpy import linalg
import math
import networkx
import plotly
import pandas
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# global variables

COUNT_NODES = 9
COUNT_BRANCHES = 17
X_COORDINATES_NODES = [2,5,25,45,5,25,45,5,25,45]
Y_COORDINATES_NODES = [7,5,5,5,25,25,25,45,45,45]

# initialization nondirected graph
def func_initialization_nondirected_adjacency_list(count_nodes, count_branches):
    graph = networkx.Graph()
    for nodes in range(count_nodes):
        graph.add_node(nodes)
    nondirected_adjacency_list= numpy.array([(1,3,5,7,9),
                                             (0,2,4),
                                             (1,3,5),
                                             (0,2,6),
                                             (1,5,7),
                                             (0,2,4,6,8),
                                             (3,5,9),
                                             (0,4,8),
                                             (5,7,9),
                                             (0,6,8)])
    for start in range(len(nondirected_adjacency_list)):
        for stop in range(len(nondirected_adjacency_list[start])):
            graph.add_edge(start, nondirected_adjacency_list[start][stop])
    return graph
def func_initialization_nondirected_adjacency_matrix(count_nodes, count_branches):
    graph = networkx.Graph()
    for nodes in range(count_nodes):
        graph.add_node(nodes)
    nondirected_adjacency_matrix = numpy.array([(0,1,0,1,0,1,0,1,0,1),
                                                (1,0,1,0,1,0,0,0,0,0),
                                                (0,1,0,1,0,1,0,0,0,0),
                                                (1,0,1,0,0,0,1,0,0,0),
                                                (0,1,0,0,0,1,0,1,0,0),
                                                (1,0,1,0,1,0,1,0,1,0),
                                                (0,0,0,1,0,1,0,0,0,1),
                                                (1,0,0,0,1,0,0,0,1,0),
                                                (0,0,0,0,0,1,0,1,0,1),
                                                (1,0,0,0,0,0,1,0,1,0)])
    for start in range(len(nondirected_adjacency_matrix)):
        for stop in range(len(nondirected_adjacency_matrix[start])):
                if(nondirected_adjacency_matrix[start][stop]):
                    graph.add_edge(start, stop)
    return graph

# initialization directed graph
def func_initialization_directed_adjacency_list(count_nodes, count_branches):
    graph = networkx.Graph()
    for nodes in range(count_nodes):
        graph.add_node(nodes)
    directed_adjacency_list= numpy.array([(1),
                                          (2,4),
                                          (3,5),
                                          (0,6),
                                          (5,7),
                                          (0,6,8),
                                          (9),
                                          (0,8),
                                          (9),
                                          (0)])
    for start in range(len(directed_adjacency_list)):
        for stop in range(len(directed_adjacency_list[start])):
            graph.add_edge(start, directed_adjacency_list[start][stop])
    return graph
# NOW JUST WORKING WITH THIS FUNCTION
def func_initialization_directed_adjacency_matrix(count_nodes, count_branches):
    graph = networkx.Graph()
    for nodes in range(count_nodes):
        graph.add_node(nodes, pos=(X_COORDINATES_NODES[nodes],Y_COORDINATES_NODES[nodes]))
    directed_adjacency_matrix = numpy.array([(0,1,0,0,0,0,0,0,0,0),
                                             (0,0,1,0,1,0,0,0,0,0),
                                             (0,0,0,1,0,1,0,0,0,0),
                                             (1,0,0,0,0,0,1,0,0,0),
                                             (0,0,0,0,0,1,0,1,0,0),
                                             (1,0,0,0,0,0,1,0,1,0),
                                             (0,0,0,1,0,1,0,0,0,1),
                                             (1,0,0,0,0,0,0,0,1,0),
                                             (0,0,0,0,0,0,0,0,0,1),
                                             (1,0,0,0,0,0,0,0,0,0)])
    for start in range(len(directed_adjacency_matrix)):
        for stop in range(len(directed_adjacency_matrix[start])):
                if(directed_adjacency_matrix[start][stop]):
                    graph.add_edge(start, stop)
    return graph

def func_count_equation(graph):
    count_equation = numpy.array()
    graph

def func_add_edge(source, goal, graph=None):
    graph.add_edge(source, goal)
    graph.add_edge(goal, source)

def func_visualization(graph):
    X_COORDINATES_NODES = [2,5,25,45,5,25,45,5,25,45]
    Y_COORDINATES_NODES = [7,5,5,5,25,25,25,45,45,45]
    pos = networkx.get_node_attributes(graph1,'pos')

    nodes_trace = go.Scatter(x=X_COORDINATES_NODES,
                             y=Y_COORDINATES_NODES,
                             hoverinfo='text',
                             mode='markers',
                             marker=dict(
                                 showscale=True,
                                 colorscale='electric',
                                 reversescale=True,
                                 color='tan',
                                 size=35,
                                 colorbar=dict(thickness=15,
                                               title='Node Connection',
                                               xanchor='left',
                                               titleside='right')
                                 ),
                             line_width=2
                             )
    x_edges = []
    y_edges = []

    for edges in graph.edges():
        x0, y0 = graph.nodes[edges[0]][pos[edges][0]]
        x1, y1 = graph.nodes[edges[1]][pos[edges][1]]
        x_edges.append(x0)
        x_edges.append(x1)
        x_edges.append(None)
        y_edges.append(y0)
        y_edges.append(y1)
        y_edges.append(None)

    edges_trace = go.Scatter(x=x_edges,
                             y=y_edges,
                             line=dict(width=0.5, color='#888'),
                             hoverinfo='none',
                             mode='lines')
    fig = go.Figure(data=[edges_trace,nodes_trace],
            layout = go.Layout(
                title='Graph',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(text="Ammotations",
                                  showarrow=False,
                                  xref="paper",
                                  yref="paper",
                                  x=0.005,
                                  y=-0.0025
                    )],
                xaxis=dict(showgrid=False, zeroline=False, showticklables=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklables=False),
                ))
    fig.show()

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


print("graph1 - graph")
graph1 = func_initialization_directed_adjacency_matrix(COUNT_NODES, COUNT_BRANCHES)
#print(graph1)
#for edges in graph1.edges():
#    print(graph1.edges())
#    print(edges)
#    print(graph1.nodes[edges[0]]['pos'])
#    print(graph1.nodes[edges[1]]['pos'])
















# visualization

# 2 strings below drawing graph by standart libraries
#networkx.draw(graph1, with_labels = True)
#plt.show()

#func_visualization(graph1)
#networkx.draw(graph1, pos)
#plt.show()

print("G - graph")
G = networkx.random_geometric_graph(10, 0.125)
print(G)
for edges in G.edges():
    print(G.edges())
    print(edges)
    print(G.nodes[edges[0]]['pos'])
    print(G.nodes[edges[1]]['pos'])
