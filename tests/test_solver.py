import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.solver import TSPLKHSolver, TSPConcordeSolver, KaMISSolver


##############################################
#             Test Func For TSP              #
##############################################

def _test_tsp_lkh_solver():
    tsp_lkh_solver  = TSPLKHSolver(lkh_max_trials=100)
    tsp_lkh_solver.from_txt("tests/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=True)
    _, gap_avg, _ = tsp_lkh_solver.evaluate(caculate_gap=True)
    print(f"TSPLKHSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = "The average gap ({gap_avg}) of TSP50 solved by TSPLKHSolver " 
        message += "is larger than or equal to 1e-2%."
        raise ValueError(message)


def _test_tsp_concorde_solver():
    tsp_lkh_solver  = TSPConcordeSolver()
    tsp_lkh_solver.from_txt("tests/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=True)
    _, gap_avg, _ = tsp_lkh_solver.evaluate(caculate_gap=True)
    print(f"TSPConcordeSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = f"The average gap ({gap_avg}) of TSP50 solved by TSPConcordeSolver " 
        message += "is larger than or equal to 1e-3%."
        raise ValueError(message)


def test_tsp():
    """
    Test TSPSolver
    """
    _test_tsp_lkh_solver()
    _test_tsp_concorde_solver()


##############################################
#            Test Func For KaMIS             #
##############################################

def _test_kamis_solver():
    kamis_solver = KaMISSolver(time_limit=20)
    kamis_solver.solve(src="tests/mis_test", out="tests/mis_test/solve")
    kamis_solver.from_folder("tests/mis_test")
    kamis_solver.from_satlib_pickle("tests/mis_test.pickle")
    gap_avg = kamis_solver.evaluate(caculate_gap=True)["avg_gap"]
    print(f"KaMISSolver Gap: {gap_avg}")
    if gap_avg >= 0.1:
        message = f"The average gap ({gap_avg}) of MIS solved by KaMISSolver " 
        message += "is larger than or equal to 0.1%."
        raise ValueError(message)    


def test_mis():
    """
    Test MISSolver
    """
    _test_kamis_solver()

##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_tsp()
    test_mis()