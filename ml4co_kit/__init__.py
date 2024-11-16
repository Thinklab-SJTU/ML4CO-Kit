import importlib.util

#######################################################
#                       Support                       #
#######################################################
from .utils.type_utils import TASK_TYPE, SOLVER_TYPE, TASK_SUPPORT_SOLVER 

#######################################################
#                      Algorithm                      #
#######################################################
from .algorithm import atsp_greedy_decoder, atsp_2opt_local_search
from .algorithm import (
    tsp_greedy_decoder, tsp_insertion_decoder, tsp_mcts_decoder, tsp_mcts_local_search
)

#######################################################
#                     Free Dataset                    #
#######################################################
from .data import VRPLIBOriDataset, CVRPUniformDataset
from .data import SATLIBOriDataset
from .data import TSPLIBOriDataset, TSPUniformDataset, TSPLIB4MLDataset, ML4TSPDataset

#######################################################
#                      Evaluator                      #
#######################################################
from .evaluate import ATSPEvaluator
from .evaluate import CVRPEvaluator, CVRPUniformEvaluator
from .evaluate import SATLIBEvaluator
from .evaluate import TSPEvaluator, TSPLIBOriEvaluator, TSPLIB4MLEvaluator, TSPUniformEvaluator

#######################################################
#                    Data Generator                   #
#######################################################
from .generator import ATSPDataGenerator
from .generator import CVRPDataGenerator
from .generator import MClDataGenerator
from .generator import MCutDataGenerator
from .generator import MISDataGenerator
from .generator import MVCDataGenerator
from .generator import TSPDataGenerator

#######################################################
#                        Solver                       #
#######################################################
from .solver import ATSPSolver, ATSPLKHSolver
from .solver import CVRPSolver, CVRPPyVRPSolver, CVRPLKHSolver, CVRPHGSSolver
from .solver import MClSolver, MClGurobiSolver
from .solver import MCutSolver, MCutGurobiSolver
from .solver import MISSolver, KaMISSolver, MISGurobiSolver
from .solver import MVCSolver, MVCGurobiSolver
from .solver import (
    TSPSolver, TSPLKHSolver, TSPConcordeSolver, 
    TSPConcordeLargeSolver, TSPGAEAXSolver, TSPGAEAXLargeSolver
)

#######################################################
#                    Utils Function                   #
#######################################################
from .utils import download, compress_folder, extract_archive, _get_md5
from .utils import iterative_execution_for_file, iterative_execution, Timer
from .utils import np_dense_to_sparse, np_sparse_to_dense, GraphData, tsplib95
from .utils import MISGraphData, MVCGraphData, MClGraphData, MCutGraphData
from .utils import sat_to_mis_graph, cnf_folder_to_gpickle_folder, cnf_to_gpickle

#######################################################
#           Extension Function (matplotlib)           #
#######################################################
found_matplotlib = importlib.util.find_spec("matplotlib")
if found_matplotlib is not None:
    from .draw.cvrp import draw_cvrp_problem, draw_cvrp_solution
    from .draw.mcl import draw_mcl_problem, draw_mcl_solution
    from .draw.mcut import draw_mcut_problem, draw_mcut_solution
    from .draw.mis import draw_mis_problem, draw_mis_solution
    from .draw.mvc import draw_mvc_problem, draw_mvc_solution
    from .draw.tsp import draw_tsp_problem, draw_tsp_solution

#######################################################
#        Extension Function (pytorch_lightning)       #
#######################################################
found_pytorch_lightning = importlib.util.find_spec("pytorch_lightning")
if found_pytorch_lightning is not None:
    from .learning.env import BaseEnv
    from .learning.model import BaseModel
    from .learning.train import Checkpoint, Logger, Trainer
    from .learning.utils import to_numpy, to_tensor
    from .learning.utils import check_dim
    from .learning.utils import points_to_distmat, sparse_points


__version__ = "0.2.2"
__author__ = "SJTU-ReThinkLab"
