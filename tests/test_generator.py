import os
import shutil
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit import TSPDataGenerator, MISDataGenerator, CVRPDataGenerator
from ml4co_kit import KaMISSolver, CVRPPyVRPSolver


##############################################
#             Test Func For TSP              #
##############################################

def _test_tsp_lkh_generator(
    num_threads: int, nodes_num: int, data_type: str, 
    regret: bool, re_download: bool=False
):
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
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
        regret=regret,
    )
    if re_download:
        tsp_data_lkh.download_lkh()
    # generate data
    tsp_data_lkh.generate()
    # remove the save path
    shutil.rmtree(save_path)


def _test_tsp_concorde_generator(
    num_threads: int, nodes_num: int, data_type: str,
    recompile_concorde: bool = False
):
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
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
    )
    if recompile_concorde:
        tsp_data_concorde.recompile_concorde()
        
    # generate data
    tsp_data_concorde.generate()
    # remove the save path
    shutil.rmtree(save_path)


def test_tsp():
    """
    Test TSPDataGenerator
    """
    # re-download lkh
    _test_tsp_lkh_generator(
        num_threads=4, nodes_num=50, data_type="uniform", 
        regret=False, re_download=True
    )
    # regret & threads
    _test_tsp_lkh_generator(
        num_threads=1, nodes_num=50, data_type="uniform", regret=True
    )
    _test_tsp_lkh_generator(
        num_threads=4, nodes_num=50, data_type="uniform", regret=True
    )
    # concorde
    _test_tsp_concorde_generator(
        num_threads=4, nodes_num=50, data_type="uniform", 
        recompile_concorde=True
    )
    # gaussian & cluster
    _test_tsp_concorde_generator(
        num_threads=4, nodes_num=50, data_type="gaussian"
    )
    _test_tsp_concorde_generator(
        num_threads=4, nodes_num=50, data_type="cluster"
    )


##############################################
#             Test Func For MIS              #
##############################################

def _test_mis_kamis(
    nodes_num_min: int, nodes_num_max: int, data_type: str,
    recompile_kamis: bool = False
):
    """
    Test MISDataGenerator using KaMIS
    """
    # save path
    save_path = f"tmp/mis_{data_type}_kamis"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create MISDataGenerator using KaMIS solver
    solver = KaMISSolver(time_limit=10)
    if recompile_kamis:
        solver.recompile_kamis()
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


def _test_mis_gurobi(
    nodes_num_min: int, nodes_num_max: int, data_type: str
):
    """
    Test MISDataGenerator using MISGurobi
    """
    # save path
    save_path = f"tmp/mis_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create MISDataGenerator using gurobi solver
    mis_data_gurobi = MISDataGenerator(
        nodes_num_min=nodes_num_min,
        nodes_num_max=nodes_num_max,
        data_type=data_type,
        solver="gurobi",
        train_samples_num=2,
        val_samples_num=2,
        test_samples_num=2,
        save_path=save_path,
        solve_limit_time=10.0,
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
    _test_mis_kamis(
        nodes_num_min=600, nodes_num_max=700, data_type="er", recompile_kamis=True
    )
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
#             Test Func For CVRP             #
##############################################

def _test_cvrp_pyvrp_generator(
    num_threads: int, nodes_num: int, data_type: str, capacity: int
):
    """
    Test CVRPDataGenerator using PyVRP
    """
    # save path
    save_path = f"tmp/cvrp_{data_type}_pyvrp"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create CVRPDataGenerator using PyVRP solver
    solver = CVRPPyVRPSolver(time_limit=3)
    cvrp_data_pyvrp = CVRPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver=solver,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
        min_capacity=capacity,
        max_capacity=capacity
    )
    # generate data
    cvrp_data_pyvrp.generate()
    # remove the save path
    shutil.rmtree(save_path)


def _test_cvrp_lkh_generator(
    num_threads: int, nodes_num: int, data_type: str, capacity: int
):
    """
    Test CVRPDataGenerator using LKH
    """
    # save path
    save_path = f"tmp/cvrp_{data_type}_pyvrp"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create CVRPDataGenerator using lkh solver
    cvrp_data_pyvrp = CVRPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver="lkh",
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
        min_capacity=capacity,
        max_capacity=capacity
    )
    # generate data
    cvrp_data_pyvrp.generate()
    # remove the save path
    shutil.rmtree(save_path)


def test_cvrp():
    """
    Test CVRPDataGenerator
    """
    # threads
    _test_cvrp_pyvrp_generator(
        num_threads=1, nodes_num=50, data_type="uniform", capacity=40
    )
    _test_cvrp_pyvrp_generator(
        num_threads=4, nodes_num=50, data_type="uniform", capacity=40
    )
    # gaussian
    _test_cvrp_pyvrp_generator(
        num_threads=4, nodes_num=50, data_type="gaussian", capacity=40
    )
    # lkh
    _test_cvrp_lkh_generator(
        num_threads=4, nodes_num=50, data_type="uniform", capacity=40
    )
    
##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_tsp()
    test_mis()
    test_cvrp()
    shutil.rmtree("tmp")
