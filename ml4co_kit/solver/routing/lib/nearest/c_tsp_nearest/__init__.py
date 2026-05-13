import os
import shutil
import pathlib


try:
    from .source import tsp_nearest_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import tsp_nearest_impl


pybind11_tsp_nearest_impl = tsp_nearest_impl.tsp_nearest
