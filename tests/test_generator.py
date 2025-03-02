import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import shutil
from ml4co_kit import *

GUROBI_TEST = False


##############################################
#             Test Func For ATSP             #
##############################################

def _test_atsp_lkh_generator(
    num_threads: int, nodes_num: int, data_type: str, sat_vars_num: int = None, 
    sat_clauses_nums: int = None, re_download: bool = False
):
    """
    Test ATSPDataGenerator using ATSPLKHSolver
    """
    # save path
    save_path = f"tmp/atsp{nodes_num}_lkh"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # create TSPDataGenerator using lkh solver
    atsp_data_lkh = ATSPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver=SOLVER_TYPE.LKH,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
        sat_vars_nums=sat_vars_num,
        sat_clauses_nums=sat_clauses_nums,
    )

    if re_download:
        atsp_data_lkh.download_lkh()    

    # generate data
    atsp_data_lkh.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def test_atsp():
    """
    Test ATSPDataGenerator
    """
    # uniform
    _test_atsp_lkh_generator(
        num_threads=4, nodes_num=50, data_type="uniform", re_download=True
    )
    # sat
    _test_atsp_lkh_generator(
        num_threads=4, nodes_num=55, data_type="sat", sat_clauses_nums=5, sat_vars_num=5
    )
    # threads
    _test_atsp_lkh_generator(
        num_threads=1, nodes_num=55, data_type="sat", sat_clauses_nums=5, sat_vars_num=5
    )
    # hcp
    _test_atsp_lkh_generator(
        num_threads=4, nodes_num=50, data_type="hcp"
    )
    

##############################################
#             Test Func For CVRP             #
##############################################

def _test_cvrp_pyvrp_generator(
    num_threads: int, nodes_num: int, data_type: str, capacity: int
):
    """
    Test CVRPDataGenerator using CVRPPyVRPSolver
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
    save_path = f"tmp/cvrp_{data_type}_lkh"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # create CVRPDataGenerator using lkh solver
    solver = CVRPLKHSolver(lkh_max_trials=100)
    cvrp_data_lkh = CVRPDataGenerator(
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
    cvrp_data_lkh.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def _test_cvrp_hgs_generator(
    num_threads: int, nodes_num: int, data_type: str, capacity: int
):
    """
    Test CVRPDataGenerator using CVRPHGSSolver
    """
    # save path
    save_path = f"tmp/cvrp_{data_type}_pyvrp"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # create CVRPDataGenerator using lkh solver
    cvrp_data_hgs = CVRPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver=SOLVER_TYPE.HGS,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
        min_capacity=capacity,
        max_capacity=capacity
    )
    
    # generate data
    cvrp_data_hgs.generate()
    
    # remove the save path
    shutil.rmtree(save_path)
    

def test_cvrp():
    """
    Test CVRPDataGenerator
    """
    # hgs
    _test_cvrp_hgs_generator(
        num_threads=4, nodes_num=50, data_type="uniform", capacity=40
    )
    # threads
    _test_cvrp_hgs_generator(
        num_threads=1, nodes_num=50, data_type="uniform", capacity=40
    )
    # lkh
    _test_cvrp_lkh_generator(
        num_threads=4, nodes_num=20, data_type="uniform", capacity=30
    )
    # pyvrp
    _test_cvrp_pyvrp_generator(
        num_threads=1, nodes_num=50, data_type="uniform", capacity=40
    )
    # gaussian
    _test_cvrp_pyvrp_generator(
        num_threads=4, nodes_num=50, data_type="gaussian", capacity=40
    )


##############################################
#             Test Func For LP               #
##############################################

def _test_lp_gurobi(num_threads: int, data_type: str):
    """
    Test LPDataGenerator using LPGurobiSolver
    """
    if not GUROBI_TEST:
        return
    
    # save path
    save_path = f"tmp/lp_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # solver
    solver = LPGurobiSolver(time_limit=1.0)
 
    # create LPDataGenerator using Gurobi solver
    lp_data_pyvrp = LPDataGenerator(
        num_threads=num_threads,
        vars_num=20,
        constr_num=16,
        sparse_ratio=0.0,
        data_type=data_type,
        solver=solver,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path
    )
    
    # generate data
    lp_data_pyvrp.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def test_lp():
    """
    Test LPDataGenerator
    """
    _test_lp_gurobi(num_threads=1, data_type="uniform")
    _test_lp_gurobi(num_threads=4, data_type="uniform")


##############################################
#             Test Func For MCl              #
##############################################

def _test_mcl_gurobi_generator(
    nodes_num_min: int, nodes_num_max: int, data_type: str
):
    """
    Test MClDataGenerator using MClGurobiSolver
    """
    if not GUROBI_TEST:
        return
    
    # save path
    save_path = f"tmp/mcl_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # solver
    solver = MClGurobiSolver(time_limit=5.0)
      
    # create MClDataGenerator using gurobi solver
    mcl_data_gurobi = MClDataGenerator(
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
    mcl_data_gurobi.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def test_mcl():
    """
    Test MCLDataGenerator
    """
    _test_mcl_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="er")
    _test_mcl_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ba")
    _test_mcl_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="hk")
    _test_mcl_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ws")


##############################################
#            Test Func For MCut              #
##############################################

def _test_mcut_gurobi_generator(
    nodes_num_min: int, nodes_num_max: int, data_type: str
):
    """
    Test MCDataGenerator using MCGurobiSolver
    """
    if not GUROBI_TEST:
        return
    
    # save path
    save_path = f"tmp/mcut_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # solver
    solver = MCutGurobiSolver(time_limit=5.0)
      
    # create MCutDataGenerator using gurobi solver
    mcut_data_gurobi = MCutDataGenerator(
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
    mcut_data_gurobi.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def test_mcut():
    """
    Test MVCDataGenerator
    """
    _test_mcut_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="er")
    _test_mcut_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ba")
    _test_mcut_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="hk")
    _test_mcut_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ws")


##############################################
#             Test Func For MIS              #
##############################################

def _test_mis_kamis_generator(
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
    solver = KaMISSolver(time_limit=1.0)
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
    
    # remove the save path
    shutil.rmtree(save_path)


def _test_mis_gurobi_generator(
    nodes_num_min: int, nodes_num_max: int, data_type: str
):
    """
    Test MISDataGenerator using MISGurobiSolver
    """
    if not GUROBI_TEST:
        return
    
    # save path
    save_path = f"tmp/mis_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # solver
    solver = MISGurobiSolver(time_limit=5.0)
      
    # create MISDataGenerator using gurobi solver
    mis_data_gurobi = MISDataGenerator(
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
    mis_data_gurobi.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def test_mis():
    """
    Test MISDataGenerator
    """
    _test_mis_kamis_generator(
        nodes_num_min=50, nodes_num_max=100, data_type="er", recompile_kamis=True
    )
    _test_mis_kamis_generator(nodes_num_min=50, nodes_num_max=100, data_type="ba")
    _test_mis_kamis_generator(nodes_num_min=50, nodes_num_max=100, data_type="hk")
    _test_mis_kamis_generator(nodes_num_min=50, nodes_num_max=100, data_type="ws")
    
    _test_mis_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="er")
    _test_mis_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ba")
    _test_mis_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="hk")
    _test_mis_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ws")


##############################################
#             Test Func For MVC              #
##############################################

def _test_mvc_gurobi_generator(
    nodes_num_min: int, nodes_num_max: int, data_type: str
):
    """
    Test MVCDataGenerator using MVCGurobiSolver
    """
    if not GUROBI_TEST:
        return
    
    # save path
    save_path = f"tmp/mvc_{data_type}_gurobi"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # solver
    solver = MVCGurobiSolver(time_limit=5.0)
      
    # create MVCDataGenerator using gurobi solver
    mvc_data_gurobi = MVCDataGenerator(
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
    mvc_data_gurobi.generate()
    
    # remove the save path
    shutil.rmtree(save_path)


def test_mvc():
    """
    Test MVCDataGenerator
    """
    _test_mvc_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="er")
    _test_mvc_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ba")
    _test_mvc_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="hk")
    _test_mvc_gurobi_generator(nodes_num_min=50, nodes_num_max=100, data_type="ws")

   
##############################################
#             Test Func For TSP              #
##############################################

def _test_tsp_lkh_generator(
    num_threads: int, nodes_num: int, data_type: str, regret: bool
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
        solver=SOLVER_TYPE.LKH,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
        regret=regret,
    )

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
    # create TSPDataGenerator using concorde solver
    tsp_data_concorde = TSPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver=SOLVER_TYPE.CONCORDE,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
    )
    if recompile_concorde:
        tsp_data_concorde._recompile_concorde()
        
    # generate data
    tsp_data_concorde.generate()
    # remove the save path
    shutil.rmtree(save_path)


def _test_tsp_ga_eax_generator(
    num_threads: int, nodes_num: int, data_type: str
):
    """
    Test TSPDataGenerator using TSPGAEAXSolver
    """
    # save path
    save_path = f"tmp/tsp{nodes_num}_ga_eax"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create TSPDataGenerator using ga-eax solver
    tsp_data_ga_eax = TSPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver=SOLVER_TYPE.GA_EAX,
        train_samples_num=4,
        val_samples_num=4,
        test_samples_num=4,
        save_path=save_path,
    )
        
    # generate data
    tsp_data_ga_eax.generate()
    # remove the save path
    shutil.rmtree(save_path)
    
    
def _test_tsp_ga_eax_large_generator(
    num_threads: int, nodes_num: int, data_type: str
):
    """
    Test TSPDataGenerator using TSPGAEAXLargeSolver
    """
    # save path
    save_path = f"tmp/tsp{nodes_num}_ga_eax_large"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create TSPDataGenerator using ga-eax-large solver
    tsp_data_ga_eax_large = TSPDataGenerator(
        num_threads=num_threads,
        nodes_num=nodes_num,
        data_type=data_type,
        solver=SOLVER_TYPE.GA_EAX_LARGE,
        train_samples_num=1,
        val_samples_num=0,
        test_samples_num=0,
        save_path=save_path,
    )
        
    # generate data
    tsp_data_ga_eax_large.generate()
    # remove the save path
    shutil.rmtree(save_path)
    
 
def test_tsp():
    """
    Test TSPDataGenerator
    """
    # threads
    _test_tsp_lkh_generator(
        num_threads=4, nodes_num=50, data_type="uniform", regret=False
    )
    # regret & threads
    _test_tsp_lkh_generator(
        num_threads=1, nodes_num=10, data_type="uniform", regret=True
    )
    _test_tsp_lkh_generator(
        num_threads=4, nodes_num=10, data_type="uniform", regret=True
    )
    # concorde
    _test_tsp_concorde_generator(
        num_threads=4, nodes_num=50, data_type="uniform", 
        recompile_concorde=True
    )
    # ga-eax
    _test_tsp_ga_eax_generator(
        num_threads=4, nodes_num=50, data_type="uniform"
    )
    # ga-eax-large
    _test_tsp_ga_eax_large_generator(
        num_threads=1, nodes_num=1000, data_type="uniform"
    )
    # gaussian & cluster
    _test_tsp_concorde_generator(
        num_threads=4, nodes_num=50, data_type="gaussian"
    )
    _test_tsp_concorde_generator(
        num_threads=4, nodes_num=50, data_type="cluster"
    )
   
    
##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_atsp()
    test_cvrp()
    test_lp()
    test_mcl()
    test_mcut()
    test_mis()
    test_mvc()
    test_tsp()
    shutil.rmtree("tmp")
