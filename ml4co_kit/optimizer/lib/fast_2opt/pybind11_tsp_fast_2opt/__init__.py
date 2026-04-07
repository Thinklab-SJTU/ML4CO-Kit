import os
import shutil
import pathlib

try:
    from .source import tsp_fast_2opt_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import tsp_fast_2opt_impl

c_tsp_fast_2opt_impl = tsp_fast_2opt_impl.fast_two_opt_local_search