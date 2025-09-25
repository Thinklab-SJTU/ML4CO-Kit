import importlib.util

###################################################
#                      Task                       #
###################################################

# Base Task
from .task import TaskBase, TASK_TYPE

# Graph Task
from .task import GraphTaskBase, MClTask, MCutTask, MISTask, MVCTask

# Routing Task
from .task import ATSPTask, CVRPTask, TSPTask, OPTask, PCTSPTask, SPCTSPTask


###################################################
#                    Generator                    #
###################################################

# Base Generator
from .generator import GeneratorBase

# Graph Generator
from .generator import (
    GraphGeneratorBase, GRAPH_TYPE, GRAPH_WEIGHT_TYPE, 
    MClGenerator, MCutGenerator, MISGenerator, MVCGenerator
)

# Routing Generator
from .generator import (
    RoutingGenerator, ATSPGenerator, CVRPGenerator, TSPGenerator,
    OPGenerator, PCTSPGenerator, SPCTSPGenerator, ATSP_TYPE, 
    CVRP_TYPE, TSP_TYPE, OP_TYPE, PCTSP_TYPE, SPCTSP_TYPE
)


####################################################
#                      Solver                      #
####################################################
# Base Solver
from .solver import SolverBase, SOLVER_TYPE

# Solver (not use torch backend)
from .solver import (
    LKHSolver, ConcordeSolver, KaMISSolver, RLSASolver, HGSSolver,
    GpDegreeSolver, LcDegreeSolver, MCTSSolver, InsertionSolver, GAEAXSolver
)

# Greedy Solver (use torch backend)
found_torch = importlib.util.find_spec("torch")
if found_torch is not None:
    from .solver import GreedySolver


####################################################
#                     Wrapper                      #
####################################################
from .wrapper import (
    WrapperBase, TSPWrapper, ATSPWrapper, CVRPWrapper,
    MISWrapper, MCutWrapper, MClWrapper, MVCWrapper,
    OPWrapper, PCTSPWrapper, SPCTSPWrapper
)


####################################################
#                  Utils Function                  #
####################################################

# File Utils
from .utils import (
    download, pull_file_from_huggingface, get_md5,
    compress_folder, extract_archive, check_file_path
)

# Time Utils
from .utils import tqdm_by_time, Timer

# GNN4CO
from .extension.gnn4co import (
    GNN4COEnv, GNN4COModel, GNNEncoder, TSPGNNEncoder
)


__version__ = "1.0.0"
__author__ = "SJTU-ReThinkLab"