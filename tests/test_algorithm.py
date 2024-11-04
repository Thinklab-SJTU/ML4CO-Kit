import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import numpy as np
from ml4co_kit import *


##############################################
#             Test Func For ATSP             #
##############################################

def test_atsp_2opt_local_search():
    solver = ATSPSolver()
    solver.from_txt("tests/data_for_tests/algorithm/atsp/atsp50.txt", ref=True)
    dists = solver.dists
    heatmap = np.load("tests/data_for_tests/algorithm/atsp/atsp50_heatmap.npy", allow_pickle=True)
    greedy_tours = atsp_greedy_decoder(heatmap=-heatmap)
    tours = atsp_2opt_local_search(init_tours=greedy_tours, dists=dists)
    solver.from_data(tours=tours, ref=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of ATSP using Greedy Decoder with 2OPT Local Search: {gap_avg}")


def test_atsp():
    test_atsp_2opt_local_search()
    
    
##############################################
#             Test Func For TSP              #
##############################################

def test_tsp_greedy_decoder():
    solver = TSPSolver()
    solver.from_txt("tests/data_for_tests/algorithm/tsp/tsp50.txt", ref=True)
    heatmap = np.load("tests/data_for_tests/algorithm/tsp/tsp50_heatmap.npy", allow_pickle=True)
    tours = tsp_greedy_decoder(heatmap=heatmap)
    solver.from_data(tours=tours, ref=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of TSP using Greedy Decoder: {gap_avg}")
    if (gap_avg-1.28114) >= 1e-5:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by Greedy Decoder "
            "is not equal to 1.28114%."
        )
        raise ValueError(message)


def test_tsp_insertion_decoder():
    solver = TSPSolver()
    solver.from_txt("tests/data_for_tests/algorithm/tsp/tsp50.txt", ref=True)
    points = solver.points
    tours = tsp_insertion_decoder(points=points)
    solver.from_data(tours=tours, ref=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of TSP using Insertion Decoder: {gap_avg}")


def test_tsp_mcts_decoder():
    solver = TSPSolver()
    solver.from_txt("tests/data_for_tests/algorithm/tsp/tsp50.txt", ref=True)
    points = solver.points
    heatmap = np.load("tests/data_for_tests/algorithm/tsp/tsp50_heatmap.npy", allow_pickle=True)
    tours = tsp_mcts_decoder(heatmap=heatmap, points=points, time_limit=0.1)
    solver.from_data(tours=tours, ref=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of TSP using MCTS Decoder : {gap_avg}")
    if gap_avg >= 1e-1:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by MCTS Decoder "
            "is larger than or equal to 1e-1%."
        )
        raise ValueError(message)
    

def test_tsp_mcts_local_search():
    solver = TSPSolver()
    solver.from_txt("tests/data_for_tests/algorithm/tsp/tsp50.txt", ref=True)
    points = solver.points
    heatmap = np.load("tests/data_for_tests/algorithm/tsp/tsp50_heatmap.npy", allow_pickle=True)
    greedy_tours = tsp_greedy_decoder(heatmap=heatmap)
    tours = tsp_mcts_local_search(
        init_tours=greedy_tours, heatmap=heatmap, points=points, time_limit=0.1
    )
    solver.from_data(tours=tours, ref=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of TSP using Greedy Decoder with MCTS Local Search: {gap_avg}")
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
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_atsp()
    test_tsp()