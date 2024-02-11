import os
import shutil

try:
    from .concorde.tsp import TSPSolver as TSPConSolver
except:
    ori_dir = os.getcwd()
    os.chdir('solver/tsp/pyconcorde')
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    shutil.rmtree("solver/tsp/pyconcorde/build")
    from .concorde.tsp import TSPSolver as TSPConSolver

    