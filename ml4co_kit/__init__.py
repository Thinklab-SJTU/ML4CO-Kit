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

# expand
found_matplotlib = importlib.util.find_spec("matplotlib")
if found_matplotlib is not None:
    from .draw.tsp import draw_tsp_problem, draw_tsp_solution
    from .draw.mis import draw_mis_problem, draw_mis_solution
    from .draw.cvrp import draw_cvrp_problem, draw_cvrp_solution
else:
    print("matplotlib not installed")


__version__ = "0.0.2a1"
__author__ = "SJTU-ReThinkLab"
