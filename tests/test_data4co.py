import os
import shutil
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co import TSPDataGenerator, MISDataGenerator, TSPLKHSolver, TSPConcordeSolver, KaMIS

##############################################
#             Test Func For TSP              #
##############################################

def _test_tsp_lkh_generator(num_threads: int, nodes_num: int, data_type: str):
    """
    Test TSPDataGenerator using LKH Solver
    """
    # save path
    save_path = f"tmp/tsp{nodes_num}_lkh"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create TSPDataGenerator using lkh solver 
    tsp_data_lkh = TSPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver="lkh",
        train_samples_num=16,
        val_samples_num=16,
        test_samples_num=16,
        save_path=save_path
    )
    # generate data
    tsp_data_lkh.generate()
    # remove the save path
    shutil.rmtree(save_path)


def _test_tsp_lkh_solver():
    tsp_lkh_solver  = TSPLKHSolver(lkh_max_trials=100)
    tsp_lkh_solver.from_txt("tests/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=True)
    _, gap_avg, _ = tsp_lkh_solver.evaluate(caculate_gap=True)
    print(f"TSPLKHSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = "The average gap ({gap_avg}) of TSP50 solved by TSPLKHSolver " 
        message += "is larger than or equal to 1e-2."
        raise ValueError(message)

def _test_tsp_concorde_generator(num_threads: int, nodes_num: int, data_type: str):
    """
    Test TSPDataGenerator using Concorde Solver
    """
    # save path
    save_path = f"tmp/tsp{nodes_num}_concorde"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create TSPDataGenerator using lkh solver 
    tsp_data_concorde = TSPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver="concorde",
        train_samples_num=16,
        val_samples_num=16,
        test_samples_num=16,
        save_path=save_path
    )
    # generate data
    tsp_data_concorde.generate()
    # remove the save path
    shutil.rmtree(save_path)


def _test_tsp_concorde_solver():
    tsp_lkh_solver  = TSPConcordeSolver()
    tsp_lkh_solver.from_txt("tests/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=True)
    _, gap_avg, _ = tsp_lkh_solver.evaluate(caculate_gap=True)
    print(f"TSPConcordeSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = f"The average gap ({gap_avg}) of TSP50 solved by TSPConcordeSolver " 
        message += "is larger than or equal to 1e-2."
        raise ValueError(message)


def test_tsp():
    """
    Test TSPDataGenerator
    """
    _test_tsp_lkh_generator(num_threads=16, nodes_num=50, data_type="uniform")
    _test_tsp_lkh_generator(num_threads=16, nodes_num=50, data_type="gaussian")
    _test_tsp_lkh_solver()
    _test_tsp_concorde_generator(num_threads=16, nodes_num=50, data_type="uniform")
    _test_tsp_concorde_generator(num_threads=16, nodes_num=50, data_type="gaussian")
    _test_tsp_concorde_solver()

##############################################
#             Test Func For MIS              #
##############################################

def _test_mis_kamis(nodes_num_min: int, nodes_num_max: int, data_type: str):
    """
    Test MISDataGenerator using KaMIS
    """
    # save path
    save_path = f"tmp/mis_{data_type}_kamis"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create TSPDataGenerator using lkh solver
    solver = KaMIS(time_limit=10)
    mis_data_kamis = MISDataGenerator(
        nodes_num_min=nodes_num_min, 
        nodes_num_max=nodes_num_max,
        data_type=data_type,
        solver=solver,
        train_samples_num=2,
        val_samples_num=2,
        test_samples_num=2,
        save_path=save_path,
    )
    # generate and solve data
    mis_data_kamis.generate()
    mis_data_kamis.solve()
    # remove the save path
    shutil.rmtree(save_path)


def _test_mis_gurobi(nodes_num_min: int, nodes_num_max: int, data_type: str):
    """
    Test MISDataGenerator using MISGurobi
    """
    # save path
    save_path = f"tmp/mis_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create TSPDataGenerator using lkh solver 
    mis_data_gurobi = MISDataGenerator(
        nodes_num_min=nodes_num_min, 
        nodes_num_max=nodes_num_max,
        data_type=data_type,
        solver="gurobi",
        train_samples_num=2,
        val_samples_num=2,
        test_samples_num=2,
        save_path=save_path,
        solve_limit_time=10.0
    )
    # generate and solve data
    mis_data_gurobi.generate()
    mis_data_gurobi.solve()
    # remove the save path
    shutil.rmtree(save_path)
    

def test_mis():
    """
    Test MISDataGenerator
    """
    _test_mis_kamis(nodes_num_min=600, nodes_num_max=700, data_type="er")
    _test_mis_kamis(nodes_num_min=600, nodes_num_max=700, data_type="ba")
    _test_mis_kamis(nodes_num_min=600, nodes_num_max=700, data_type="hk")
    _test_mis_kamis(nodes_num_min=600, nodes_num_max=700, data_type="ws")
    # gurobi need license
    # gurobipy.GurobiError: Model too large for size-limited license;
    # visit https://www.gurobi.com/free-trial for a full license
    # _test_mis_gurobi(nodes_num_min=600, nodes_num_max=700, data_type="er")
    # _test_mis_gurobi(nodes_num_min=600, nodes_num_max=700, data_type="ba")
    # _test_mis_gurobi(nodes_num_min=600, nodes_num_max=700, data_type="hk")
    # _test_mis_gurobi(nodes_num_min=600, nodes_num_max=700, data_type="ws")

##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_tsp()
    test_mis()
    shutil.rmtree("tmp")