import os
import pathlib

try:
    from .source import cython_tsp_greedy
except:
    cython_merge_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(cython_merge_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)    
    from .source import cython_tsp_greedy

cython_tsp_greedy = cython_tsp_greedy.cython_tsp_greedy