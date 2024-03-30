from data4co.evaluate import TSPLIBOriginEvaluator
from data4co.solver import TSPLKHSolver, TSPConcordeSolver

lkh_solver = TSPLIBOriginEvaluator()
eva = TSPLIBOriginEvaluator()
eva.evaluate(lkh_solver, norm="EUC_2D")