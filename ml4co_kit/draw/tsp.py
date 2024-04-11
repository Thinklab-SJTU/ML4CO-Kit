import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union


def edges_to_node_pairs(edge_target: np.ndarray):
    """Helper function to convert edge matrix into pairs of adjacent nodes."""
    pairs = []
    for r in range(len(edge_target)):
        for c in range(len(edge_target)):
            if edge_target[r][c] == 1:
                pairs.append((r, c))
    return pairs


def draw_tsp_problem(
    save_path: str,
    points: Union[list, np.ndarray],
    edge_values: np.ndarray = None,
    figsize: tuple = (5, 5),
    node_color: str = "darkblue",
    edge_color: str = "darkblue",
    node_size: int = 50,
):
    # check
    if "." not in save_path:
        save_path += ".png"
    if type(points) == list:
        points = np.array(points)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2:
        raise ValueError("the dim of the points must 2.")

    # edge_values
    if edge_values is None:
        edge_values = (
            np.sum(
                (np.expand_dims(points, 1) - np.expand_dims(points, 0)) ** 2, axis=-1
            )
            ** 0.5
        )

    # edge_target
    nodes_num = points.shape[0]
    edge_target = np.zeros((nodes_num, nodes_num))
    target_pairs = edges_to_node_pairs(edge_target)
    graph = nx.from_numpy_array(edge_values)
    pos = dict(zip(range(len(points)), points.tolist()))

    # plt
    figure = plt.figure(figsize=figsize)
    figure.add_subplot(111)
    nx.draw_networkx_nodes(G=graph, pos=pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(
        G=graph, pos=pos, edgelist=target_pairs, alpha=1, width=1, edge_color=edge_color
    )

    # save
    plt.savefig(save_path)


def draw_tsp_solution(
    save_path: str,
    points: Union[list, np.ndarray],
    tours: Union[list, np.ndarray],
    edge_values: np.ndarray = None,
    figsize: tuple = (5, 5),
    node_color: str = "darkblue",
    edge_color: str = "darkblue",
    node_size: int = 50,
):
    # check
    if "." not in save_path:
        save_path += ".png"
    if type(points) == list:
        points = np.array(points)
    if type(tours) == list:
        tours = np.array(tours)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if tours.ndim == 2 and tours.shape[0] == 1:
        tours = tours[0]
    if points.ndim != 2:
        raise ValueError("the dim of the points must 2.")
    if tours.ndim != 1:
        raise ValueError("the dim of the tours must 1.")

    # edge_values
    if edge_values is None:
        edge_values = (
            np.sum(
                (np.expand_dims(points, 1) - np.expand_dims(points, 0)) ** 2, axis=-1
            )
            ** 0.5
        )

    # edge_target
    nodes_num = points.shape[0]
    edge_target = np.zeros((nodes_num, nodes_num))
    for i in range(len(tours) - 1):
        edge_target[tours[i], tours[i + 1]] = 1
    target_pairs = edges_to_node_pairs(edge_target)
    graph = nx.from_numpy_array(edge_values)
    pos = dict(zip(range(len(points)), points.tolist()))

    # plt
    figure = plt.figure(figsize=figsize)
    figure.add_subplot(111)
    nx.draw_networkx_nodes(G=graph, pos=pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(
        G=graph, pos=pos, edgelist=target_pairs, alpha=1, width=1, edge_color=edge_color
    )

    # save
    plt.savefig(save_path)
