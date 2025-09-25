import os
import pathlib


GA_EAX_LARGE_BASE_PATH = pathlib.Path(__file__).parent
GA_EAX_LARGE_SOLVER_PATH = pathlib.Path(__file__).parent / "ga_eax_large_solver"

# Determining whether the solvers have been built
if not os.path.exists(GA_EAX_LARGE_SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(GA_EAX_LARGE_BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)