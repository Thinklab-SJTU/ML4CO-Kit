"""
Draw the problem and the solution to TSP, and save to the specficed path.  
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union


def edges_to_node_pairs(edge_target: np.ndarray):
    r"""
    Helper function to convert edge matrix into pairs of adjacent nodes.
    """
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
    r"""
    the method to draw the problem of tsp.
    
    :param save_path: string, the path to save figure as png format.
    :param points: list or np.ndarry, the points coordinates data.
    :param edge_values: no.ndarry, the weights of edges.
    :param figsize: tuple, the size of the image, defaults to (5,5).
    :param node_color: string, the color of the node, defaults to darkblue.
    :param edge_color: string, the color of the edge, defaults to darkblue.
    :param node_size: int, the size of the coordinates, defaults to 50.
    
    ..dropdown:: Example
    
        ::

            >>> from ml4co_kit import TSPSolver, draw_tsp_problem
            # create TSPSolver
            >>> solver = TSPSolver()
            
            # load data from the tsplib folder
            >>> solver.from_tsplib_folder(
                    tsp_folder_path="examples/tsp/tsplib_2/problem",
                    tour_folder_path="examples/tsp/tsplib_2/solution"
                )
            
            >>> draw_tsp_problem("./data/img",solver.point)
            # the image will be stored in the specified path if the process is successful.
   
    """
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
    r"""
    the method to draw the solution to tsp.
    
    :param save_path: string, the path to save figure as png format.
    :param points: list or np.ndarry, the points coordinates data.
    :param tour: list or np.ndarry, the solution tour of the problem.
    :param edge_values: no.ndarry, the weights of edges.
    :param figsize: tuple, the size of the image, defaults to (5,5).
    :param node_color: string, the color of the node, defaults to darkblue.
    :param edge_color: string, the color of the edge, defaults to darkblue.
    :param node_size: int, the size of the coordinates, defaults to 50.
    
    ..dropdown:: Example
    
        ::

            >>> from ml4co_kit import TSPSolver, draw_tsp_solution
            # create TSPSolver
            >>> solver = TSPSolver()
            
            # load data from the tsplib folder
            >>> solver.from_tsplib_folder(
                    tsp_folder_path="examples/tsp/tsplib_2/problem",
                    tour_folder_path="examples/tsp/tsplib_2/solution"
                )
            
            >>> draw_tsp_problem("./data/img",solver.point,solver.tours)
            # the image will be stored in the specified path if the process is successful.
    """
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
