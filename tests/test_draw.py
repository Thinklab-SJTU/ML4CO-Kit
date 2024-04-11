import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit.draw.tsp import draw_tsp_solution, draw_tsp_problem
from ml4co_kit.draw.mis import draw_mis_solution, draw_mis_problem

from ml4co_kit.solver import TSPConcordeSolver, KaMISSolver


def test_draw_tsp():
    solver = TSPConcordeSolver(scale=100)
    solver.from_tsp("tests/draw_test/eil101.tsp")
    solver.solve()
    draw_tsp_problem(
        save_path="tests/draw_test/eil101_problem.png",
        points=solver.ori_points,
    )
    draw_tsp_solution(
        save_path="tests/draw_test/eil101_solution.png",
        points=solver.ori_points,
        tours=solver.tours,
    )


def test_mis_tsp():
    # use KaMISSolver to solve the problem
    mis_solver = KaMISSolver()
    mis_solver.solve(src="tests/draw_test/mis_example")

    # draw
    draw_mis_problem(
        save_path="tests/draw_test/mis_problem.png", 
        gpickle_path="tests/draw_test/mis_example/mis_example.gpickle"
    )
    draw_mis_solution(
        save_path="tests/draw_test/mis_solution.png",
        gpickle_path="tests/draw_test/mis_example/mis_example.gpickle",
        result_path="tests/draw_test/mis_example/solve/mis_example_unweighted.result"
    )
    
    
if __name__ == "__main__":
    test_draw_tsp()
    test_mis_tsp()
