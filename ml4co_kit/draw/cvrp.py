r"""
Draw the problem and the solution to CVRP, and save to the specficed path.  
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
import matplotlib.pyplot as plt
from typing import Union


def draw_cvrp_problem(
    save_path: str,
    depots: Union[list, np.ndarray],
    points: Union[list, np.ndarray],
    figsize: tuple = (8, 8),
    node_size: int = 50
):
    r"""
    the method to draw the problem of cvrp.
    
    :param save_path: string, the path to save figure as png format.
    :param depots: list or np.ndarry, the depots coordinates data.
    :param points: list or np.ndarry, the customer points coordinates data.
    :param figsize: tuple, the size of the image, defaults to (8,8).
    :param node_size: int, the size of the coordinates, defaults to 50.
    
    .. note::
        the dimension of ``depots`` must be 1, and the dimension of ``points`` must be 2.
        
    ..dropdown:: Example
    
        ::
            >>> from ml4co_kit import CVRPSolver,draw_cvrp_problem
                
            # create CVRPSolver
            >>> solver = CVRPSolver()

            # load data from ``.txt`` file
            >>> solver.from_txt(file_path="examples/cvrp/txt/cvrp20_hgs_1s_6.13013.txt")

            >>> draw_cvrp_problem("./data/img",solver.depots,solver.points)
            # the image will be stored in the specified path if the process is successful.
    """
    # check
    if "." not in save_path:
        save_path += ".png"
    if type(depots) == list:
        depots = np.array(depots)
    if depots.ndim == 2 and depots.shape[0] == 1:
        depots = depots[0]
    if depots.ndim != 1:
        raise ValueError("the dim of the depots must 1.")
    if type(points) == list:
        points = np.array(points)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2:
        raise ValueError("the dim of the points must 2.")
    
    # plot
    _, ax = plt.subplots(figsize=figsize)
    kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
    ax.scatter(depots[0], depots[1], label="Depot", **kwargs)
    ax.scatter(points[:, 0], points[:, 1], s=node_size, label="Clients")
    ax.grid(color="grey", linestyle="solid", linewidth=0.2)
    ax.set_aspect("equal", "datalim")
    ax.legend(frameon=False, ncol=2)

    # save
    plt.savefig(save_path)
    
    
def draw_cvrp_solution(
    save_path: str,
    depots: Union[list, np.ndarray],
    points: Union[list, np.ndarray],
    tour: Union[list, np.ndarray],
    figsize: tuple = (8, 8),
    node_size: int = 50
):
    r"""
    the method to draw the solution to cvrp.
    
    :param save_path: string, the path to save figure as png format.
    :param depots: list or np.ndarry, the depots coordinates data.
    :param points: list or np.ndarry, the customer points coordinates data.
    :param tour: list or np.ndarry, the solution tour of the problem.
    :param figsize: tuple, the size of the image, defaults to (8,8).
    :param node_size: int, the size of the coordinates, defaults to 50.
    
    .. note::
        the dimension of ``depots`` must be 1, the dimension of ``points`` must be 2, and the dimension of ``tour`` must be 1.
        
    ..dropdown:: Example
    
        ::
            >>> from ml4co_kit import CVRPSolver,draw_cvrp_solution
                
            # create CVRPSolver
            >>> solver = CVRPSolver()

            # load data from ``.txt`` file
            >>> solver.from_txt(file_path="examples/cvrp/txt/cvrp20_hgs_1s_6.13013.txt")

            >>> draw_cvrp_solution("./data/img",solver.depots,solver.points,solver.tour)
            # the image will be stored in the specified path if the process is successful.
    """
    # check
    if "." not in save_path:
        save_path += ".png"
    if type(depots) == list:
        depots = np.array(depots)
    if depots.ndim == 2 and depots.shape[0] == 1:
        depots = depots[0]
    if depots.ndim != 1:
        raise ValueError("the dim of the depots must 1.")
    if type(points) == list:
        points = np.array(points)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2:
        raise ValueError("the dim of the points must 2.")
    if type(tour) == list:
        tour = np.array(tour)
    if tour.ndim == 2 and tour.shape[0] == 1:
        tour = tour[0]
    if tour.ndim != 1:
        raise ValueError("the dim of the tour must 1.")
    
    # plot
    _, ax = plt.subplots(figsize=figsize)
    kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
    ax.scatter(depots[0], depots[1], label="Depot", **kwargs)

    coords = np.concatenate([np.expand_dims(depots, axis=0), points], axis=0)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    split_tours = np.split(tour, np.where(tour == 0)[0])[1: -1]
    idx = 0
    for part_tour in split_tours:
        route = part_tour[1:]
        x = x_coords[route]
        y = y_coords[route]

        # Coordinates of clients served by this route.
        if len(route) == 1:
            ax.scatter(x, y, label=f"Route {idx}", zorder=3, s=node_size)
        ax.plot(x, y)
        arrowprops = dict(arrowstyle='->', linewidth=0.25, color='grey')
        ax.annotate(
            text='', 
            xy=(x_coords[0], y_coords[0]), 
            xytext=(x[0], y[0]), 
            arrowprops=arrowprops
        )
        ax.annotate(
            text='', 
            xy=(x[-1], y[-1]), 
            xytext=(x_coords[0], y_coords[0]), 
            arrowprops=arrowprops
        )
    
    ax.set_aspect("equal", "datalim")
    ax.legend(frameon=False, ncol=2)

    # save
    plt.savefig(save_path)