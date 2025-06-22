import importlib.util

#######################################
#             ATSP Solver             #  
#######################################
from .atsp.base import ATSPSolver
from .atsp.lkh import ATSPLKHSolver
from .atsp.or_tools import ATSPORSolver

#######################################
#             CVRP Solver             #  
#######################################
from .cvrp.base import CVRPSolver
from .cvrp.hgs import CVRPHGSSolver
from .cvrp.lkh import CVRPLKHSolver
from .cvrp.pyvrp import CVRPPyVRPSolver

#######################################
#              KP Solver              #
#######################################
from .kp.base import KPSolver
from .kp.or_tools import KPORSolver

#######################################
#              OP Solver              #
#######################################
from .op.base import OPSolver
from .op.gurobi import OPGurobiSolver

#######################################
#              LP Solver             #  
#######################################
from .lp.base import LPSolver
from .lp.gurobi import LPGurobiSolver

#######################################
#              MCl Solver             #  
#######################################
from .mcl.base import MClSolver
from .mcl.gurobi import MClGurobiSolver
from .mcl.or_tools import MClORSolver

#######################################
#             MCut Solver             #  
#######################################
from .mcut.base import MCutSolver
from .mcut.gurobi import MCutGurobiSolver
from .mcut.or_tools import MCutORSolver

#######################################
#              MIS Solver             #  
#######################################
from .mis.base import MISSolver
from .mis.gurobi import MISGurobiSolver
from .mis.kamis import KaMISSolver
from .mis.or_tools import MISORSolver

#######################################
#              MVC Solver             #  
#######################################
from .mvc.base import MVCSolver
from .mvc.gurobi import MVCGurobiSolver
from .mvc.or_tools import MVCORSolver

#######################################
#             TSP Solver             #  
#######################################
from .tsp.base import TSPSolver
from .tsp.concorde import TSPConcordeSolver
from .tsp.concorde_large import TSPConcordeLargeSolver
from .tsp.ga_eax_normal import TSPGAEAXSolver
from .tsp.ga_eax_large import TSPGAEAXLargeSolver
from .tsp.lkh import TSPLKHSolver
from .tsp.or_tools import TSPORSolver

#######################################
#             PCTSP Solver             #  
#######################################
from .pctsp.base import PCTSPSolver
from .pctsp.or_tools import PCTSPORSolver

#######################################
#     Extension Function (torch)      #
#######################################
found_torch = importlib.util.find_spec("torch")
if found_torch is not None:
    from .tsp.neurolkh import TSPNeuroLKHSolver