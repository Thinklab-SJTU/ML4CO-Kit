import importlib.util

# base
from .data import TSPLIBOriDataset, TSPUniformDataset
from .data import SATLIBOriDataset
from .data import VRPLIBOriDataset
from .evaluate import TSPEvaluator, TSPLIBOriEvaluator, TSPUniformEvaluator
from .evaluate import SATLIBEvaluator
from .evaluate import CVRPEvaluator
from .generator import TSPDataGenerator, MISDataGenerator, CVRPDataGenerator
from .solver import TSPSolver, TSPLKHSolver, TSPConcordeSolver
from .solver import MISSolver, KaMISSolver, MISGurobi
from .solver import CVRPSolver, CVRPPyVRPSolver, CVRPLKHSolver
from .utils import download, compress_folder, extract_archive, _get_md5

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
    from .learning.model import BaseModel, ModelConfigurator
    from .learning.search import SearchConfigurator
    from .learning.train import Checkpoint, Logger, Trainer
    from .learning.utils import to_numpy, to_tensor
    from .learning.utils import check_dim
    from .learning.utils import points_to_distmat, sparse_points


__version__ = "0.0.2"
__author__ = "SJTU-ReThinkLab"
