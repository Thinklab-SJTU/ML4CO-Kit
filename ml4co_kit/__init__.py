r"""
ML4CO-Kit Module.
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


import importlib.util

###################################################
#                      Task                       #
###################################################

# Base Task
from .task import TaskBase, TASK_TYPE

# Graph Task
from .task import GraphTaskBase, Graph, GraphSetTaskBase
from .task import MClTask, MCutTask, MISTask, MVCTask, GMTask

# Routing Task
from .task import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE
from .task import ATSPTask, CVRPTask, OPTask, PCTSPTask, SPCTSPTask, TSPTask


###################################################
#                    Generator                    #
###################################################

# Base Generator
from .generator import GeneratorBase

# Graph Generator
from .generator import (
    GraphWeightGenerator, GraphGeneratorBase, 
    GRAPH_TYPE, GRAPH_WEIGHT_TYPE, 
)
from .generator import MClGenerator, MCutGenerator, MISGenerator, MVCGenerator

# Routing Generator
from .generator import RoutingGenerator
from .generator import (
    ATSP_TYPE, CVRP_TYPE, OP_TYPE, 
    PCTSP_TYPE, SPCTSP_TYPE, TSP_TYPE
)
from .generator import (
    ATSPGenerator, CVRPGenerator, OPGenerator,  
    PCTSPGenerator, SPCTSPGenerator, TSPGenerator, 
)


####################################################
#                      Solver                      #
####################################################
# Base Solver
from .solver import SolverBase, SOLVER_TYPE

# Solver (not use torch backend)
from .solver import (
    ConcordeSolver, GAEAXSolver, GpDegreeSolver, GurobiSolver, 
    HGSSolver, ILSSolver, InsertionSolver, KaMISSolver, 
    LcDegreeSolver, LKHSolver, ORSolver
)

# Solver (use torch backend)
found_torch = importlib.util.find_spec("torch")
if found_torch is not None:
    from .solver import (
        BeamSolver, GreedySolver, MCTSSolver, NeuroLKHSolver, RLSASolver
    )


####################################################
#                     Wrapper                      #
####################################################

# Base Wrapper
from .wrapper import (
    WrapperBase,
)

# Routing Problems
from .wrapper import (
    ATSPWrapper, CVRPWrapper, OPWrapper, 
    PCTSPWrapper, SPCTSPWrapper, TSPWrapper
)

# Graph Problems
from .wrapper import (
    MClWrapper, MCutWrapper, MISWrapper, MVCWrapper
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

# Type Utils
from .utils import to_numpy, to_tensor

# GNN4CO
from .extension.gnn4co import (
    GNN4COEnv, GNN4COModel, GNNEncoder, TSPGNNEncoder
)


__version__ = "1.0.0"
__author__ = "SJTU-ReThinkLab"