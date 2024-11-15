import os
import pathlib


GA_EAX_NORMAL_BASE_PATH = pathlib.Path(__file__).parent
GA_EAX_NORMAL_SOLVER_PATH = pathlib.Path(__file__).parent / "ga_eax_normal_solver"
GA_EAX_NORMAL_TMP_PATH = pathlib.Path(__file__).parent / "tmp"

# Determining whether the solvers have been built
if not os.path.exists(GA_EAX_NORMAL_SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(GA_EAX_NORMAL_BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)


# make tmp dir
if not os.path.exists(GA_EAX_NORMAL_TMP_PATH):
    os.makedirs(GA_EAX_NORMAL_TMP_PATH)


def tsp_ga_eax_normal_solve(
    max_trials: int, sol_name: str, population_num: int,
    offspring_num: int, tsp_name: str, show_info: bool = False
):
    show_info = 1 if show_info else 0
    tsp_path = os.path.join("tmp", tsp_name)
    sol_path = os.path.join("tmp", sol_name)
    ori_dir = os.getcwd()
    os.chdir(GA_EAX_NORMAL_BASE_PATH)
    command = f"./ga_eax_normal_solver {max_trials} {sol_path} {population_num} {offspring_num} {tsp_path} {show_info}"
    os.system(command)
    os.chdir(ori_dir)

    