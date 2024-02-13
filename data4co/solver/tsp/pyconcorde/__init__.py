import os
import shutil
import pathlib


try:
    from .concorde.tsp import TSPSolver as TSPConSolver
except:
    concorde_path = pathlib.Path(__file__).parent.parent / "pyconcorde"
    ori_dir = os.getcwd()
    os.chdir(concorde_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    shutil.rmtree(f"{concorde_path}/build")
    from .concorde.tsp import TSPSolver as TSPConSolver

    