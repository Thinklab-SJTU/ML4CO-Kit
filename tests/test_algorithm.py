import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import numpy as np
from ml4co_kit.algorithm import (
    tsp_greedy_decoder, tsp_insertion_decoder, tsp_mcts_decoder, 
    tsp_mcts_local_search, atsp_greedy_decoder
)
from ml4co_kit.solver import TSPSolver, ATSPSolver


##############################################
#             Test Func For TSP              #
##############################################

def test_tsp_greedy_decoder():
    solver = TSPSolver()
    solver.from_txt("tests/algorithm_test/tsp50.txt")
    points = solver.points
    heatmap = np.load("tests/algorithm_test/tsp50_heatmap.npy", allow_pickle=True)
    tours = tsp_greedy_decoder(heatmap=heatmap, points=points)
    solver.read_tours(tours)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Greedy Decoder Gap: {gap_avg}")
    if (gap_avg-1.28114) >= 1e-5:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by Greedy Decoder "
            "is not equal to 1.28114%."
        )
        raise ValueError(message)


def test_tsp_insertion_decoder():
    solver = TSPSolver()
    solver.from_txt("tests/algorithm_test/tsp50.txt")
    points = solver.points
    tours = tsp_insertion_decoder(points=points)
    solver.read_tours(tours)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Insertion Decoder Gap: {gap_avg}")


def test_tsp_mcts_decoder():
    solver = TSPSolver()
    solver.from_txt("tests/algorithm_test/tsp50.txt")
    points = solver.points
    heatmap = np.load("tests/algorithm_test/tsp50_heatmap.npy", allow_pickle=True)
    tours = tsp_mcts_decoder(heatmap=heatmap, points=points, time_limit=0.1)
    solver.read_tours(tours)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"MCTS Decoder Gap: {gap_avg}")
    if gap_avg >= 1e-1:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by MCTS Decoder "
            "is larger than or equal to 1e-1%."
        )
        raise ValueError(message)
    

def test_tsp_mcts_local_search():
    solver = TSPSolver()
    solver.from_txt("tests/algorithm_test/tsp50.txt")
    points = solver.points
    heatmap = np.load("tests/algorithm_test/tsp50_heatmap.npy", allow_pickle=True)
    greedy_tours = tsp_greedy_decoder(heatmap=heatmap, points=points)
    tours = tsp_mcts_local_search(
        init_tours=greedy_tours, heatmap=heatmap, points=points, time_limit=0.1
    )
    solver.read_tours(tours)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"MCTS Decoder Gap: {gap_avg}")
    if gap_avg >= 1e-1:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by Greedy+MCTS "
            "is larger than or equal to 1e-1%."
        )
        raise ValueError(message)
    
  
def test_tsp():
    test_tsp_greedy_decoder()
    test_tsp_insertion_decoder()
    test_tsp_mcts_decoder()
    test_tsp_mcts_local_search()
    

##############################################
#             Test Func For ATSP             #
##############################################

def test_atsp_greedy_decoder():
    solver = ATSPSolver()
    solver.from_txt("tests/solver_test/atsp50_test.txt")
    dists = solver.dists
    tours = atsp_greedy_decoder(dists)
    solver.read_tours(tours)
    costs_avg = solver.evaluate()
    print(f"Greedy Decoder Costs: {costs_avg}")
    if (costs_avg-2.39802) >= 1e-5:
        message = (
            f"The average costs ({costs_avg}) of TSP50 solved by Greedy Decoder "
            "is not equal to 2.39802."
        )
        raise ValueError(message)

def test_atsp():
    test_atsp_greedy_decoder()
    
    
##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_tsp()
    test_atsp()