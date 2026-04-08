import os
import pathlib
import shutil

try:
    from .source import lc_degree_impl
except ImportError:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import lc_degree_impl

c_mis_lc_degree = lc_degree_impl.mis_lc_degree
c_mvc_lc_degree = lc_degree_impl.mvc_lc_degree
c_mcl_lc_degree = lc_degree_impl.mcl_lc_degree
c_mcut_lc_degree = lc_degree_impl.mcut_lc_degree
