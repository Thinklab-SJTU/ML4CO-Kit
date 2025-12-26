import os
import pathlib


try:
    from .mis_mcmc_lib import mis_mcmc_enhanced_impl
except:
    root_path = pathlib.Path(__file__).parent
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("make clean")
    os.system("make")
    os.chdir(ori_dir)
    from .mis_mcmc_lib import mis_mcmc_enhanced_impl

