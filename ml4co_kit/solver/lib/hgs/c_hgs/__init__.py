import os
import pathlib


HGS_BASE_PATH = pathlib.Path(__file__).parent
HGS_SOLVER_PATH = HGS_BASE_PATH / "cvrp_hgs_solver"

# Determining whether the solvers have been built
if not os.path.exists(HGS_SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(HGS_BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)  