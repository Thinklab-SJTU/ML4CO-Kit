import os
import shutil
import pathlib


try:
    from .source import ps_gen_impl
except:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import ps_gen_impl

pybind11_ps_gen_func = ps_gen_impl.generate_ps_clauses

