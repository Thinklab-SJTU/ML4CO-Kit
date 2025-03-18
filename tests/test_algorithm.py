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
    print(f"Gap of ATSP using Greedy Decoder with 2OPT Local Search: {gap_avg}%")


def test_atsp():
    test_atsp_2opt_local_search()
    

##############################################
#             Test Func For MCl              #
##############################################

def test_mcl_greedy_decoder():
    solver = MClSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcl/mcl_rb_small.txt", ref=True)
    heatmap = np.load("tests/data_for_tests/algorithm/mcl/mcl_rb_small_heatmap.npy", allow_pickle=True)
    solver.graph_data[0].nodes_label = mcl_greedy_decoder(
        heatmap=heatmap, graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCl using Greedy Decoder: {gap_avg}%")


def test_mcl_beam_decoder():
    solver = MClSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcl/mcl_rb_small.txt", ref=True)
    heatmap = np.load("tests/data_for_tests/algorithm/mcl/mcl_rb_small_heatmap.npy", allow_pickle=True)
    solver.graph_data[0].nodes_label = mcl_beam_decoder(
        heatmap=heatmap, graph=solver.graph_data[0].to_matrix(), beam_size=16
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCl using Beam Decoder: {gap_avg}%")


def test_mcl_gp_degree_decoder():
    solver = MClSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcl/mcl_rb_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mcl_gp_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCl using GP-Degree Decoder: {gap_avg}%")


def test_mcl_lc_degree_decoder():
    solver = MClSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcl/mcl_rb_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mcl_lc_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCl using LC-Degree Decoder: {gap_avg}%")
    
    
def test_mcl_rlsa_decoder():
    solver = MClSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcl/mcl_rb_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        sols.append(
            mcl_rlsa_decoder(graph=graph.to_matrix())
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCl using RLSA Decoder: {gap_avg}%")
    

def test_mcl_rlsa_local_search():
    solver = MClSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcl/mcl_rb_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        adj_matrix = graph.to_matrix()
        lc_sol = mcl_gp_degree_decoder(graph=adj_matrix)
        sols.append(
            mcl_rlsa_local_search(init_sol=lc_sol, graph=adj_matrix)
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCl using GP-Degree Deocder with RLSA LocalSearch: {gap_avg}%")


def test_mcl():
    test_mcl_greedy_decoder()
    test_mcl_beam_decoder()
    test_mcl_gp_degree_decoder()
    test_mcl_lc_degree_decoder()
    test_mcl_rlsa_decoder()
    test_mcl_rlsa_local_search()
    

##############################################
#             Test Func For MCut             #
##############################################

def test_mcut_lc_degree_decoder():
    solver = MCutSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcut/mcut_ba_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mcut_lc_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCut using LC-Degree Decoder: {gap_avg}%")


def test_mcut_rlsa_decoder():
    solver = MCutSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcut/mcut_ba_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        sols.append(
            mcut_rlsa_decoder(graph=graph.to_matrix(), edge_index=graph.edge_index)
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCut using RLSA Decoder: {gap_avg}%")
    

def test_mcut_rlsa_local_search():
    solver = MCutSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mcut/mcut_ba_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        adj_matrix = graph.to_matrix()
        lc_sol = mcut_lc_degree_decoder(graph=adj_matrix)
        sols.append(mcut_rlsa_local_search(
            init_sol=lc_sol, graph=adj_matrix, edge_index=graph.edge_index
        ))
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MCut using LC-Degree Deocder with RLSA LocalSearch: {gap_avg}%")
    
    
def test_mcut():
    test_mcut_lc_degree_decoder()
    test_mcut_rlsa_decoder()
    test_mcut_rlsa_local_search()
    
    
##############################################
#             Test Func For MIS              #
##############################################

def test_mis_greedy_decoder():
    solver = MISSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mis/mis_rb_small.txt", ref=True)
    heatmap = np.load("tests/data_for_tests/algorithm/mis/mis_rb_small_heatmap.npy", allow_pickle=True)
    solver.graph_data[0].nodes_label = mis_greedy_decoder(
        heatmap=heatmap, graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MIS using Greedy Decoder: {gap_avg}%")


def test_mis_beam_decoder():
    solver = MISSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mis/mis_rb_small.txt", ref=True)
    heatmap = np.load("tests/data_for_tests/algorithm/mis/mis_rb_small_heatmap.npy", allow_pickle=True)
    solver.graph_data[0].nodes_label = mis_beam_decoder(
        heatmap=heatmap, graph=solver.graph_data[0].to_matrix(), beam_size=16
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MIS using Beam Decoder: {gap_avg}%")


def test_mis_gp_degree_decoder():
    solver = MISSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mis/mis_rb_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mis_gp_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MIS using GP-Degree Decoder: {gap_avg}%")


def test_mis_lc_degree_decoder():
    solver = MISSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mis/mis_rb_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mis_lc_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MIS using LC-Degree Decoder: {gap_avg}%")
    
    
def test_mis_rlsa_decoder():
    solver = MISSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mis/mis_rb_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        sols.append(
            mis_rlsa_decoder(graph=graph.to_matrix())
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MIS using RLSA Decoder: {gap_avg}%")
    

def test_mis_rlsa_local_search():
    solver = MISSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mis/mis_rb_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        adj_matrix = graph.to_matrix()
        lc_sol = mis_gp_degree_decoder(graph=adj_matrix)
        sols.append(
            mis_rlsa_local_search(init_sol=lc_sol, graph=adj_matrix)
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MIS using GP-Degree Deocder with RLSA LocalSearch: {gap_avg}%")


def test_mis():
    test_mis_greedy_decoder()
    test_mis_beam_decoder()
    test_mis_gp_degree_decoder()
    test_mis_lc_degree_decoder()
    test_mis_rlsa_decoder()
    test_mis_rlsa_local_search()


##############################################
#             Test Func For MVC              #
##############################################

def test_mvc_greedy_decoder():
    solver = MVCSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mvc/mvc_rb_small.txt", ref=True)
    heatmap = np.load("tests/data_for_tests/algorithm/mvc/mvc_rb_small_heatmap.npy", allow_pickle=True)
    solver.graph_data[0].nodes_label = mvc_greedy_decoder(
        heatmap=heatmap, graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MVC using Greedy Decoder: {gap_avg}%")


def test_mvc_gp_degree_decoder():
    solver = MVCSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mvc/mvc_rb_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mvc_gp_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MVC using GP-Degree Decoder: {gap_avg}%")


def test_mvc_lc_degree_decoder():
    solver = MVCSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mvc/mvc_rb_small.txt", ref=True)
    solver.graph_data[0].nodes_label = mvc_lc_degree_decoder(
        graph=solver.graph_data[0].to_matrix()
    )
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MVC using LC-Degree Decoder: {gap_avg}%")
    
    
def test_mvc_rlsa_decoder():
    solver = MVCSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mvc/mvc_rb_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        sols.append(
            mvc_rlsa_decoder(graph=graph.to_matrix())
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MVC using RLSA Decoder: {gap_avg}%")
    

def test_mvc_rlsa_local_search():
    solver = MVCSolver()
    solver.from_txt("tests/data_for_tests/algorithm/mvc/mvc_rb_small_4.txt", ref=True)
    sols = list()
    for graph in solver.graph_data:
        adj_matrix = graph.to_matrix()
        lc_sol = mvc_gp_degree_decoder(graph=adj_matrix)
        sols.append(
            mvc_rlsa_local_search(init_sol=lc_sol, graph=adj_matrix)
        )
    solver.from_graph_data(nodes_label=sols, cover=False)
    _, _, gap_avg, _ = solver.evaluate(calculate_gap=True)
    print(f"Gap of MVC using GP-Degree Deocder with RLSA LocalSearch: {gap_avg}%")


def test_mvc():
    test_mvc_greedy_decoder()
    test_mvc_gp_degree_decoder()
    test_mvc_lc_degree_decoder()
    test_mvc_rlsa_decoder()
    test_mvc_rlsa_local_search()
    
    
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
    test_mcl()
    test_mcut()
    test_mis()
    test_mvc()
    test_tsp()