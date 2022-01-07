import numpy as numpy   # просто нумапай
import pandas as pd      # просто пандас
import plotly           # библиотека для визуализации
import plotly.express as px     # библиотека для визуализации
import plotly.graph_objs as go  # для визуализации

from pyvis.network import Network
import networkx as nx       # для визупализации графов и сетей

edges_x = [4, 4, None, 4, 54, None, 54, 54 , None, 4, 54, None, 4, 54, None, 29, 29, None]
edges_y = [4, 54, None, 4, 4, None, 4, 54, None, 54, 54, None, 29, 29, None, 4, 54, None]

edges_trace = go.Scatter(x=edges_x, 
                         y=edges_y, 
                         line=dict(width=0.5, color='#888'),
                         hoverinfo='none',
                         mode='lines')

nodes_x = [1]
nodes_y = [1]


nodes_trace = go.Scatter(x=nodes_x,
                         y=nodes_y,
                         mode='markers',
                         hoverinfo='text',
                         marker=dict(
                             showscale=True,
                             colorscale='Electric',
                             reversescale=True,
                             color=[],
                             size=10,
                             colorbar=dict(
                                 thickness=15,
                                 title='Node Connections',
                                 xanchor='left',
                                 titleside='right'
                                 )
                             ),
                         line_width=2)

fig = go.Figure(data=[edges_trace, nodes_trace],
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
# э литтл бит
