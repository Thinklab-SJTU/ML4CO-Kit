import pickle
import networkx as nx
import matplotlib.pyplot as plt
from ml4co_kit.draw.utils import get_pos_layer


def draw_mis_problem(
    save_path: str,
    gpickle_path: str,
    undirected: bool = True,
    self_loop: bool = False,
    pos_type: str = "kamada_kawai_layout",
    figsize: tuple = (5, 5),
    node_color: str = "darkblue",
    edge_color: str = "darkblue",
    node_size: int = 50,
):
    # check the save path
    if "." not in save_path:
        save_path += ".png"

    # read data from .gpickle
    with open(gpickle_path, "rb") as f:
        graph = pickle.load(f)

    # check if undirected/self_loop
    graph: nx.Graph
    if undirected:
        graph = graph.to_undirected()
    self_loop_edges = list(nx.selfloop_edges(graph))
    if self_loop:
        graph.add_edges_from(self_loop_edges)
    else:
        graph.remove_edges_from(self_loop_edges)

    # pos
    pos_layer = get_pos_layer(pos_type)
    pos = pos_layer(graph)

    # plt
    figure = plt.figure(figsize=figsize)
    figure.add_subplot(111)
    nx.draw_networkx_nodes(G=graph, pos=pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(
        G=graph, pos=pos, edgelist=graph.edges, alpha=1, width=1, edge_color=edge_color
    )

    # save
    plt.savefig(save_path)


def draw_mis_solution(
    save_path: str,
    gpickle_path: str,
    result_path: str,
    undirected: bool = True,
    self_loop: bool = False,
    pos_type: str = "kamada_kawai_layout",
    figsize: tuple = (5, 5),
    sel_node_color: str = "orange",
    unsel_node_color: str = "darkblue",
    edge_color: str = "darkblue",
    node_size: int = 50,
):
    # check the save path
    if "." not in save_path:
        save_path += ".png"

    # read data from .gpickle
    with open(gpickle_path, "rb") as f:
        graph = pickle.load(f)

    # read result from .result
    with open(result_path, "r") as f:
        node_labels = [int(_) for _ in f.read().splitlines()]

    # check if undirected/self_loop
    graph: nx.Graph
    if undirected:
        graph = graph.to_undirected()
    self_loop_edges = list(nx.selfloop_edges(graph))
    if self_loop:
        graph.add_edges_from(self_loop_edges)
    else:
        graph.remove_edges_from(self_loop_edges)

    # pos
    pos_layer = get_pos_layer(pos_type)
    pos = pos_layer(graph)

    # plt
    figure = plt.figure(figsize=figsize)
    figure.add_subplot(111)
    colors = [unsel_node_color if bit == 0 else sel_node_color for bit in node_labels]
    nx.draw_networkx_nodes(G=graph, pos=pos, node_color=colors, node_size=node_size)
    nx.draw_networkx_edges(
        G=graph, pos=pos, edgelist=graph.edges, alpha=1, width=1, edge_color=edge_color
    )

    # save
    plt.savefig(save_path)
