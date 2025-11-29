import os
import ctypes
import pathlib
import platform

os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_atsp_2opt_path = pathlib.Path(__file__).parent
    c_atsp_2opt_so_path = pathlib.Path(__file__).parent / "atsp_2opt.so"
    try:
        lib = ctypes.CDLL(c_atsp_2opt_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(c_atsp_2opt_path)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(c_atsp_2opt_so_path)
    ctypes_atsp_2opt_ls_impl = lib.atsp_2opt_local_search
    ctypes_atsp_2opt_ls_impl.argtypes = [
        ctypes.POINTER(ctypes.c_short), # tour
        ctypes.POINTER(ctypes.c_float), # dists
        ctypes.c_int,                   # nodes_num
        ctypes.c_int,                   # max_iterations_2opt
    ]
    ctypes_atsp_2opt_ls_impl.restype = ctypes.POINTER(ctypes.c_int)
