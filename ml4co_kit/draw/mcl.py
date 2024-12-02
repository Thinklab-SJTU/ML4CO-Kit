"""
draw_mcl
"""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ml4co_kit.draw.utils import get_pos_layer
from ml4co_kit.utils.graph.mcl import MClGraphData


def draw_mcl_problem(
    save_path: str,
    graph_data: MClGraphData,
    self_loop: bool = None,
    pos_type: str = "kamada_kawai_layout",
    figsize: tuple = (5, 5),
    node_color: str = "darkblue",
    node_size: int = 20,
    edge_color: str = "darkblue",
    edge_alpha: float = 0.5,
    edge_width: float = 1.0,   
):
    # to networkx
    nx_graph: nx.Graph = graph_data.to_networkx()

    # self loop
    if self_loop is not None:
        self_loop_edges = list(nx.selfloop_edges(nx_graph))
        if self_loop == True:
            nx_graph.add_edges_from(self_loop_edges)
        else:
            nx_graph.remove_edges_from(self_loop_edges)
            
    # pos
    pos_layer = get_pos_layer(pos_type)
    pos = pos_layer(nx_graph)

    # plt
    figure = plt.figure(figsize=figsize)
    figure.add_subplot(111)
    nx.draw_networkx_nodes(
        G=nx_graph, pos=pos, node_color=node_color, node_size=node_size
    )
    nx.draw_networkx_edges(
        G=nx_graph, pos=pos, edgelist=nx_graph.edges, 
        alpha=edge_alpha, width=edge_width, edge_color=edge_color
    )

    # save
    plt.savefig(save_path)


def draw_mcl_solution(
    save_path: str,
    graph_data: MClGraphData,
    self_loop: bool = None,
    pos_type: str = "kamada_kawai_layout",
    figsize: tuple = (5, 5),
    sel_node_color: str = "orange",
    unsel_node_color: str = "darkblue",
    node_size: int = 20,
    clique_edge_color: str = "darkblue",
    clique_edge_alpha: float = 1.0,
    clique_edge_width: float = 1.0,
    other_edge_color: str = "gray",
    other_edge_alpha: float = 0.5,
    other_edge_width: float = 0.5,
):
    # Convert to networkx graph
    nx_graph: nx.Graph = graph_data.to_networkx()
    nodes_label = graph_data.nodes_label

    # Self-loop handling
    if self_loop is not None:
        self_loop_edges = list(nx.selfloop_edges(nx_graph))
        if self_loop:
            nx_graph.add_edges_from(self_loop_edges)
        else:
            nx_graph.remove_edges_from(self_loop_edges)

    # Get position layout for nodes
    pos_layer = get_pos_layer(pos_type)
    pos = pos_layer(nx_graph)

    # Prepare plot
    figure = plt.figure(figsize=figsize)
    figure.add_subplot(111)
    
    # Node colors based on selection
    colors = [unsel_node_color if bit == 0 else sel_node_color for bit in nodes_label]
    nx.draw_networkx_nodes(G=nx_graph, pos=pos, node_color=colors, node_size=node_size)
    
    # Separate edges into clique and non-clique edges
    clique_nodes = set(np.where(nodes_label == 1)[0])  # Nodes in the maximum clique
    clique_edges = [(u, v) for u, v in nx_graph.edges if u in clique_nodes and v in clique_nodes]
    other_edges = [(u, v) for u, v in nx_graph.edges if (u, v) not in clique_edges]

    # Draw edges in different colors
    nx.draw_networkx_edges(
        G=nx_graph, pos=pos, edgelist=clique_edges, alpha=clique_edge_alpha, 
        width=clique_edge_width, edge_color=clique_edge_color
    )
    nx.draw_networkx_edges(
        G=nx_graph, pos=pos, edgelist=other_edges, alpha=other_edge_alpha, 
        width=other_edge_width, edge_color=other_edge_color
    )

    # Save the plot
    plt.savefig(save_path)
