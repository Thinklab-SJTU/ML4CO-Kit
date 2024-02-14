import numpy as np
from scipy.spatial.distance import cdist
import os
import tsplib95
import scipy.sparse
import scipy.spatial   


#############################################
##                Evaluator                ##
#############################################

class TSPEvaluator(object):
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost


#############################################
##               Write Function            ##
#############################################

def generate_tsp_file(node_coords:np.ndarray, filename):
    """
    Generate a TSP problem data file based on the given point node_coords.

    Args:
        node_coords: A two-dimensional list containing the point node_coords, e.g. [[x1, y1], [x2, y2], ...]
        filename: The filename of the generated TSP problem data file

    """
    if node_coords.ndim == 3:
        shape = node_coords.shape
        if shape[0] == 1:
            node_coords = node_coords.squeeze(axis=0)
            _generate_tsp_file(node_coords,filename)
        else:
            for i in range(shape[0]):
                _filename = filename + '-' + str(i) 
                _generate_tsp_file(node_coords[i],_filename)
    else:
        assert node_coords.ndim == 2
        _generate_tsp_file(node_coords,filename)
             
             
def _generate_tsp_file(node_coords:np.ndarray, filename):
    num_points = node_coords.shape[0]
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_points):
            x, y = node_coords[i]
            f.write(f"{i+1} {x} {y}\n")
        f.write("EOF\n")


def generate_opt_tour_file(tour:np.ndarray, filename):
    """
    Generate an opt.tour file based on the given tour.

    Args:
        tour: A one-dimensional numpy array containing the tour, e.g. [1, 5, 3, 2, 4]
        filename: The filename of the generated opt.tour file

    """
    if tour.ndim == 2:
        shape = tour.shape
        if shape[0] == 1:
            tour = tour.squeeze(axis=0)
            _generate_opt_tour_file(tour,filename)
        else:
            for i in range(shape[0]):
                _filename = filename + '-' + str(i) 
                _generate_opt_tour_file(tour[i],_filename)
    else:
        assert tour.ndim == 1
        _generate_opt_tour_file(tour,filename)
   
   
def _generate_opt_tour_file(tour:np.ndarray, filename):
    """
    Generate an opt.tour file based on the given tour.

    Args:
        tour: A one-dimensional numpy array containing the tour, e.g. [1, 5, 3, 2, 4]
        filename: The filename of the generated opt.tour file

    """
    num_points = len(tour)
    file_basename = os.path.basename(filename)
    with open(filename, 'w') as f:
        f.write(f"NAME: {file_basename}\n")
        f.write(f"TYPE: TOUR\n")
        f.write(f"DIMENSION: {num_points}\n")
        f.write(f"TOUR_SECTION\n")
        for i in range(num_points):
            f.write(f"{tour[i]}\n")
        f.write(f"-1\n")
        f.write(f"EOF\n")

    
#############################################
##         get data/tour from files        ##
#############################################

def get_data_from_tsp_file(filename: str):
    if not filename.endswith('.tsp'):
        raise ValueError("The file name must end with '. tsp'")
    tsp_data = tsplib95.load(filename)
    if tsp_data.node_coords == {}:
        num_nodes = tsp_data.dimension
        edge_weights = tsp_data.edge_weights
        edge_weights = [elem for sublst in edge_weights for elem in sublst]
        new_edge_weights = np.zeros(shape=(num_nodes,num_nodes))
        if (num_nodes * (num_nodes-1) / 2) == len(edge_weights):
            """
            [[0,1,1,1,1],
             [0,0,1,1,1],
             [0,0,0,1,1],
             [0,0,0,0,1],
             [0,0,0,0,0]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i >= j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
            new_edge_weights = new_edge_weights.T + new_edge_weights
        elif (num_nodes * (num_nodes+1) / 2) == len(edge_weights):
            """
            [[x,1,1,1,1],
             [0,x,1,1,1],
             [0,0,x,1,1],
             [0,0,0,x,1],
             [0,0,0,0,x]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i > j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
            new_edge_weights = new_edge_weights.T + new_edge_weights
        elif ((num_nodes-1) * (num_nodes-1)) == len(edge_weights):
            """
            [[0,1,1,1,1],
             [1,0,1,1,1],
             [1,0,0,1,1],
             [1,1,1,0,1],
             [1,1,1,1,0]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1
        elif (num_nodes * num_nodes) == len(edge_weights):
            """
            [[x,1,1,1,1],
             [1,x,1,1,1],
             [1,0,x,1,1],
             [1,1,1,x,1],
             [1,1,1,1,x]]
            """
            pt = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    new_edge_weights[i][j] = edge_weights[pt]
                    pt += 1  
        else:
            raise ValueError("edge_weights cannot form a Symmetric matrix") 
        return None, new_edge_weights
    else:
        node_coords = np.array(list(tsp_data.node_coords.values()))
    return node_coords, np.array(cdist(node_coords, node_coords))


def get_tour_from_tour_file(filename) -> np.ndarray:
    tsp_tour = tsplib95.load(filename)
    tsp_tour = tsp_tour.tours
    tsp_tour: list
    tsp_tour = tsp_tour[0]
    tsp_tour.append(1)
    np_tour = np.array(tsp_tour) - 1
    return np_tour
