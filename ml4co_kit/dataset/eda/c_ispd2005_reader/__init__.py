import os
import sys
import shutil
import pathlib


try:
    from .source import ispd2005_io_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system(f"{sys.executable} ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import ispd2005_io_impl


ISPD2005Reader = ispd2005_io_impl.ISPD2005Reader
