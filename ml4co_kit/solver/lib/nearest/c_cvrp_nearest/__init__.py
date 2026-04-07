import os
import shutil
import pathlib


try:
    from .source import cvrp_nearest_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import cvrp_nearest_impl


pybind11_cvrp_nearest_segment_impl = cvrp_nearest_impl.cvrp_nearest_segment
