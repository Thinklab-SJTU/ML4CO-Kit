import os
import shutil
import pathlib


try:
    from .source import mis_mcmc_lib
except:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import mis_mcmc_lib

mis_mcmc_enhanced_impl = mis_mcmc_lib.mis_mcmc_enhanced_impl

