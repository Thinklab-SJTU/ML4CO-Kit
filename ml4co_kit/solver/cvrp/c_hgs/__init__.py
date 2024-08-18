import os
import pathlib


HGS_BASE_PATH = pathlib.Path(__file__).parent
HGS_SOLVER_PATH = pathlib.Path(__file__).parent / "cvrp_hgs_solver"
HGS_TMP_PATH = pathlib.Path(__file__).parent / "tmp"

# Determining whether the solvers have been built
if not os.path.exists(HGS_SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(HGS_BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)


# make tmp dir
if not os.path.exists(HGS_TMP_PATH):
    os.makedirs(HGS_TMP_PATH)


def cvrp_hgs_solver(vrp_name: str, sol_name: str, time_limit: float, show_info: bool):
    vrp_path = os.path.join("tmp", vrp_name)
    sol_path = os.path.join("tmp", sol_name)
    ori_dir = os.getcwd()
    os.chdir(HGS_BASE_PATH)
    command = f"./cvrp_hgs_solver {vrp_path} {sol_path} -t {time_limit} "
    command += ("-si 1" if show_info else "-si 0")
    os.system(command)
    os.chdir(ori_dir)

    