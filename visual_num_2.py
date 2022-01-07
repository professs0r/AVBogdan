import numpy as numpy   # просто нумапай
import pandas as pd      # просто пандас
import plotly           # библиотека для визуализации
import plotly.express as px     # библиотека для визуализации
import plotly.graph_objs as go  # для визуализации

from pyvis.network import Network
import networkx as nx       # для визупализации графов и сетей

edges_x = [1, 4, None, 4, 4, None, 4, 54, None, 54, 54 , None, 4, 54, None, 4, 54, None, 29, 29, None, 4, 6, None, 29, 31, None, 54, 56, None, 54, 56, None]
edges_y = [1, 4, None, 4, 54, None, 4, 4, None, 4, 54, None, 54, 54, None, 29, 29, None, 4, 54, None, 54, 52, None, 29, 27, None, 54, 52, None, 4, 2, None]

edges_trace = go.Scatter(x=edges_x, 
                         y=edges_y, 
                         line=dict(width=1.5, color='grey'),
                         hoverinfo='none',
                         mode='lines')

nodes_sources_x = [1]
nodes_sources_y = [1]


nodes_sources_trace = go.Scatter(x=nodes_sources_x,
                                 y=nodes_sources_y,
                                 mode='markers',
                                 hoverinfo='text',
                                 marker=dict(
                                    showscale=True,
                                     colorscale='Electric',
                                     reversescale=True,
                                     color='tan',
                                     size=35,
                                     colorbar=dict(
                                         thickness=15,
                                         title='Node Connections',
                                         xanchor='left',
                                         titleside='right'
                                         )
                                     ),
                                 line_width=2)

nodes_TP_x = [6, 31, 56, 56]
nodes_TP_y = [52, 27, 52, 2]

nodes_TP_trace = go.Scatter(x=nodes_TP_x,
                         y=nodes_TP_y,
                         mode='markers',
                         hoverinfo='text',
                         marker=dict(
                             showscale=True,
                             colorscale='Hot',
                             reversescale=True,
                             color='teal',
                             size=25,
                             colorbar=dict(
                                 thickness=15,
                                 title='Node Connections',
                                 xanchor='left',
                                 titleside='right'
                                 )
                             ),
                         line_width=2)



fig = go.Figure(data=[edges_trace, nodes_sources_trace, nodes_TP_trace],
        layout=go.Layout(
            title='<br>Graph',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="Annotations",
                         showarrow=False,
                         xref="paper",
                         yref="paper",
                         x=0.005,
                         y=-0.002)],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )

fig.show()
