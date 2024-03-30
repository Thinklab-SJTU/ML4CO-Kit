import importlib.util

# base 
from .data import TSPLIBOriginDataset, TSPUniformDataset
from .data import SATLIBData, SATLIBDataset
from .evaluate import TSPEvaluator, TSPLIBOriginEvaluator, TSPUniformEvaluator
from .evaluate import SATLIBEvaluator
from .generator import TSPDataGenerator, MISDataGenerator
from .solver import TSPSolver, TSPLKHSolver, TSPConcordeSolver
from .solver import MISSolver, KaMISSolver, MISGurobi
from .utils import download, compress_folder, extract_archive, _get_md5

# expand
found_matplotlib = importlib.util.find_spec("matplotlib")
if found_matplotlib is not None:
    from .draw.tsp import draw_tsp_problem, draw_tsp_solution
else:
    print("matplotlib not installed")


__version__ = '0.0.1a16'
__author__ = 'ThinkLab at SJTU'