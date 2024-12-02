"""
draw_mis
"""



import networkx as nx
import matplotlib.pyplot as plt
from ml4co_kit.draw.utils import get_pos_layer
from ml4co_kit.utils.graph.mis import MISGraphData


def draw_mis_problem(
    save_path: str,
    graph_data: MISGraphData,
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


def draw_mis_solution(
    save_path: str,
    graph_data: MISGraphData,
    self_loop: bool = None,
    pos_type: str = "kamada_kawai_layout",
    figsize: tuple = (5, 5),
    sel_node_color: str = "orange",
    unsel_node_color: str = "darkblue",
    node_size: int = 20,
    edge_color: str = "darkblue",
    edge_alpha: float = 0.5,
    edge_width: float = 1.0,   
):
    # to networkx
    nx_graph: nx.Graph = graph_data.to_networkx()
    nodes_label = graph_data.nodes_label
    
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
    colors = [unsel_node_color if bit == 0 else sel_node_color for bit in nodes_label]
    nx.draw_networkx_nodes(G=nx_graph, pos=pos, node_color=colors, node_size=node_size)
    nx.draw_networkx_edges(
        G=nx_graph, pos=pos, edgelist=nx_graph.edges, alpha=edge_alpha, 
        width=edge_width, edge_color=edge_color
    )

    # save
    plt.savefig(save_path)
