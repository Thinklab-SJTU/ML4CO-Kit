import os
import shutil
import pathlib


try:
    from .source import atsp_2opt_impl
except:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import atsp_2opt_impl

pybind11_atsp_2opt_impl = atsp_2opt_impl.two_opt_local_search

