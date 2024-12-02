#######################################
#            CVRP Datasets            #  
#######################################
from .cvrp.vrplib_original import VRPLIBOriDataset
from .cvrp.cvrp_uniform import CVRPUniformDataset

#######################################
#            MIS Datasets            #  
#######################################
from .mis.satlib_original import SATLIBOriDataset

#######################################
#             TSP Datasets            #  
#######################################
from .tsp.tsplib_original import TSPLIBOriDataset
from .tsp.tsplib4ml import TSPLIB4MLDataset
from .tsp.tsp_uniform import TSPUniformDataset
from .tsp.ml4tsp import ML4TSPDataset