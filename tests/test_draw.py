import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit.draw.tsp import draw_tsp_solution, draw_tsp_problem
from ml4co_kit.draw.mis import draw_mis_solution, draw_mis_problem
from ml4co_kit.draw.cvrp import draw_cvrp_solution, draw_cvrp_problem
from ml4co_kit.solver import TSPConcordeSolver, KaMISSolver, CVRPHGSSolver


def test_draw_tsp():
    # use TSPConcordeSolver to solve the problem
    solver = TSPConcordeSolver(scale=100)
    solver.from_tsplib(tsp_file_path="tests/data_for_tests/draw/tsp/tsp_draw_example.tsp")
    solver.solve()
    
    # draw
    draw_tsp_problem(
        save_path="tests/data_for_tests/draw/tsp/tsp_draw_example_problem.png",
        points=solver.ori_points,
    )
    draw_tsp_solution(
        save_path="tests/data_for_tests/draw/tsp/tsp_draw_example_solution.png",
        points=solver.ori_points,
        tours=solver.tours,
    )


def test_draw_mis():
    # use KaMISSolver to solve the problem
    mis_solver = KaMISSolver()
    mis_solver.from_gpickle_result_folder(
        gpickle_folder_path="tests/data_for_tests/draw/mis/mis_draw_example/instance"
    )
    mis_solver.solve(
        src="tests/data_for_tests/draw/mis/mis_draw_example/instance",
        out="tests/data_for_tests/draw/mis/mis_draw_example/solution"
    )

    # draw
    draw_mis_problem(
        save_path="tests/data_for_tests/draw/mis/mis_draw_example_problem.png", 
        gpickle_path="tests/data_for_tests/draw/mis/mis_draw_example/instance/mis_draw_example.gpickle"
    )
    draw_mis_solution(
        save_path="tests/data_for_tests/draw/mis/mis_draw_example_solution.png", 
        gpickle_path="tests/data_for_tests/draw/mis/mis_draw_example/instance/mis_draw_example.gpickle",
        result_path="tests/data_for_tests/draw/mis/mis_draw_example/solution/mis_draw_example_unweighted.result"
    )


def test_draw_cvrp():
    # use CVRPHGSSolver to solve the problem
    solver = CVRPHGSSolver(depots_scale=1, points_scale=1, time_limit=1)
    solver.from_vrplib(vrp_file_path="tests/data_for_tests/draw/cvrp/cvrp_draw_example.vrp")
    solver.solve()

    # draw
    draw_cvrp_problem(
        save_path="tests/data_for_tests/draw/cvrp/cvrp_draw_example_problem.png",
        depots=solver.depots,
        points=solver.points
    )
    draw_cvrp_solution(
        save_path="tests/data_for_tests/draw/cvrp/cvrp_draw_example_solution.png",
        depots=solver.depots,
        points=solver.points,
        tour=solver.tours
    )

    
if __name__ == "__main__":
    test_draw_tsp()
    test_draw_mis()
    test_draw_cvrp()
