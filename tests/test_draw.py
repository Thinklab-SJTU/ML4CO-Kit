import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit import *


##############################################
#             Test Func For CVRP             #
##############################################

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


##############################################
#             Test Func For MCl              #
##############################################

def test_draw_mcl():
    # use MClSolver to load the data
    mcl_solver = MClSolver()
    mcl_solver.from_txt("tests/data_for_tests/draw/mcl/mcl_example.txt", ref=False)

    # draw
    draw_mcl_problem(
        graph_data=mcl_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mcl/mcl_draw_example_problem.png",
        self_loop=False 
    )
    draw_mcl_solution(
        graph_data=mcl_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mcl/mcl_draw_example_solution.png", 
        self_loop=False 
    )
    

##############################################
#             Test Func For MCut             #
##############################################

def test_draw_mcut():
    # use MCutSolver to load the data
    mcut_solver = MCutSolver()
    mcut_solver.from_txt("tests/data_for_tests/draw/mcut/mcut_example.txt", ref=False)

    # draw
    draw_mcut_problem(
        graph_data=mcut_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mcut/mcut_draw_example_problem.png",
        self_loop=False ,
    )
    draw_mcut_solution(
        graph_data=mcut_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mcut/mcut_draw_example_solution.png", 
        self_loop=False 
    )


##############################################
#             Test Func For MIS              #
##############################################

def test_draw_mis():
    # use MISSolver to load the data
    mis_solver = MISSolver()
    mis_solver.from_txt("tests/data_for_tests/draw/mis/mis_example.txt", ref=False)

    # draw
    draw_mis_problem(
        graph_data=mis_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mis/mis_draw_example_problem.png",
        self_loop=False ,
    )
    draw_mis_solution(
        graph_data=mis_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mis/mis_draw_example_solution.png", 
        self_loop=False 
    )


##############################################
#             Test Func For MVC              #
##############################################

def test_draw_mvc():
    # use MVCSolver to load the data
    mvc_solver = MVCSolver()
    mvc_solver.from_txt("tests/data_for_tests/draw/mvc/mvc_example.txt", ref=False)

    # draw
    draw_mvc_problem(
        graph_data=mvc_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mvc/mvc_draw_example_problem.png",
        self_loop=False ,
    )
    draw_mvc_solution(
        graph_data=mvc_solver.graph_data[0],
        save_path="tests/data_for_tests/draw/mvc/mvc_draw_example_solution.png", 
        self_loop=False 
    )


##############################################
#             Test Func For TSP              #
##############################################

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


##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_draw_cvrp()
    test_draw_mcl()
    test_draw_mcut()
    test_draw_mis()
    test_draw_mvc()
    test_draw_tsp()
