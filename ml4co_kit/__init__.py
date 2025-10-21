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


####################################################
#                  Utils Function                  #
####################################################

# Env Utils
from .utils import EnvInstallHelper, EnvChecker
env_checker = EnvChecker()

# File Utils
from .utils import (
    download, pull_file_from_huggingface, get_md5,
    compress_folder, extract_archive, check_file_path
)

# Time Utils
from .utils import tqdm_by_time, Timer

# Type Utils
if env_checker.check_torch():
    from .utils import to_numpy, to_tensor


###################################################
#                      Task                       #
###################################################

# Base Task
from .task import TaskBase, TASK_TYPE

# Graph Task
from .task import GraphTaskBase
from .task import MClTask, MCutTask, MISTask, MVCTask

# Routing Task
from .task import RoutingTaskBase, DISTANCE_TYPE, ROUND_TYPE
from .task import ATSPTask, CVRPTask, OPTask, PCTSPTask, SPCTSPTask, TSPTask

# Portfolio Task
from .task import PortfolioTaskBase
from .task import MinVarPOTask, MaxRetPOTask, MOPOTask


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

# Portfolio Generator
from .generator import PortfolioGeneratorBase, PO_TYPE, PortfolioDistributionArgs
from .generator import MinVarPOGenerator, MaxRetPOGenerator, MOPOGenerator

# Routing Generator
from .generator import RoutingGeneratorBase
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
    LcDegreeSolver, LKHSolver, ORSolver, SCIPSolver
)

# Solver (use torch backend)
if env_checker.check_gnn4co():
    from .solver import (
        BeamSolver, GreedySolver, MCTSSolver
    )
if env_checker.check_torch():
    from .solver import (
        NeuroLKHSolver, RLSASolver
    )


####################################################
#                    Optimizer                     #
####################################################

# Base Optimizer
from .optimizer import OptimizerBase, OPTIMIZER_TYPE

# Optimizer (not use torch backend)
from .optimizer import CVRPLSOptimizer

# Optimizer (use torch backend)
if env_checker.check_torch():
    from .optimizer import (
        TwoOptOptimizer, MCTSOptimizer, RLSAOptimizer
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

# Portfolio Problems
from .wrapper import (
    MinVarPOWrapper, MaxRetPOWrapper, MOPOWrapper
)


####################################################
#                    Learning                      #
####################################################

if env_checker.pytorch_lightning_support:
    from .learning import (
        BaseEnv, BaseModel, Trainer, Checkpoint, Logger
    )
    

####################################################
#                Version and Author                #
####################################################

__version__ = "0.4.2"
__author__ = "SJTU-ReThinkLab"