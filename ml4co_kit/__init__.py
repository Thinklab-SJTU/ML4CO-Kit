import importlib.util

# base
from .algorithm import tsp_greedy_decoder, tsp_insertion_decoder, tsp_mcts_decoder
from .algorithm import tsp_mcts_local_search
from .algorithm import atsp_greedy_decoder
from .algorithm import atsp_2opt_local_search
from .data import TSPLIBOriDataset, TSPUniformDataset, TSPLIB4MLDataset, ML4TSPDataset
from .data import SATLIBOriDataset
from .data import VRPLIBOriDataset, CVRPUniformDataset
from .evaluate import TSPEvaluator, TSPLIBOriEvaluator, TSPLIB4MLEvaluator, TSPUniformEvaluator
from .evaluate import SATLIBEvaluator
from .evaluate import CVRPEvaluator, CVRPUniformEvaluator
from .evaluate import ATSPEvaluator
from .generator import TSPDataGenerator, MISDataGenerator, CVRPDataGenerator, ATSPDataGenerator
from .solver import TSPSolver, TSPLKHSolver, TSPConcordeSolver, TSPConcordeLargeSolver
from .solver import TSPGAEAXSolver, TSPGAEAXLargeSolver
from .solver import MISSolver, KaMISSolver, MISGurobiSolver
from .solver import CVRPSolver, CVRPPyVRPSolver, CVRPLKHSolver, CVRPHGSSolver
from .solver import ATSPSolver, ATSPLKHSolver
from .utils import download, compress_folder, extract_archive, _get_md5
from .utils import iterative_execution_for_file, iterative_execution
from .utils import np_dense_to_sparse, np_sparse_to_dense, GraphData, tsplib95
from .utils import MISGraphData, sat_to_mis_graph, cnf_folder_to_gpickle_folder, cnf_to_gpickle

# expand - matplotlib
found_matplotlib = importlib.util.find_spec("matplotlib")
if found_matplotlib is not None:
    from .draw.tsp import draw_tsp_problem, draw_tsp_solution
    from .draw.mis import draw_mis_problem, draw_mis_solution
    from .draw.cvrp import draw_cvrp_problem, draw_cvrp_solution

# expand - pytorch_lightning
found_pytorch_lightning = importlib.util.find_spec("pytorch_lightning")
if found_pytorch_lightning is not None:
    from .learning.env import BaseEnv
    from .learning.model import BaseModel
    from .learning.train import Checkpoint, Logger, Trainer
    from .learning.utils import to_numpy, to_tensor
    from .learning.utils import check_dim
    from .learning.utils import points_to_distmat, sparse_points


__version__ = "0.2.0"
__author__ = "SJTU-ReThinkLab"
