#######################################
#            ATSP Evaluator           #  
#######################################
from .atsp.base import ATSPEvaluator

#######################################
#             ATSP Evaluator             #  
#######################################
from .cvrp.base import CVRPEvaluator
from .cvrp.uniform_eval import CVRPUniformEvaluator

#######################################
#            MIS Evaluator            #  
#######################################
from .mis.satlib_eval import SATLIBEvaluator

#######################################
#            TSP Evaluator            #  
#######################################
from .tsp.base import TSPEvaluator
from .tsp.tsplib_original_eval import TSPLIBOriEvaluator
from .tsp.uniform_eval import TSPUniformEvaluator
from .tsp.tsplib4ml_eval import TSPLIB4MLEvaluator
