import os
import pathlib
import shutil


try:
    from .source import mcmc_impl
except Exception:
    root_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(root_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{root_path}/build"):
        shutil.rmtree(f"{root_path}/build")
    from .source import mcmc_impl


pybind11_mis_mcmc_impl = mcmc_impl.mis_mcmc
pybind11_tsp_mcmc_impl = mcmc_impl.tsp_mcmc
pybind11_cvrp_mcmc_impl = mcmc_impl.cvrp_mcmc
