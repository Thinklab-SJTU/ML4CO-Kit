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