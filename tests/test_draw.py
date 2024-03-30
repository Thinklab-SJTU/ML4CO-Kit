import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.draw.tsp import draw_tsp_solution, draw_tsp_problem
from data4co.solver import TSPConcordeSolver


def _test_draw_tsp():
    solver = TSPConcordeSolver(concorde_scale=100)
    solver.from_tsp("tests/eil101.tsp")
    solver.solve()
    draw_tsp_problem(
        save_path="tests/eil101_problem.png",
        points=solver.ori_points,
    )
    draw_tsp_solution(
        save_path="tests/eil101_solution.png",
        points=solver.ori_points,
        tours=solver.tours,
    )


if __name__ == "__main__":
    _test_draw_tsp()  
    