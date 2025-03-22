import ctypes
import platform
import os
import pathlib

os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_cvrp_local_search = pathlib.Path(__file__).parent
    c_cvrp_local_search_so_path = pathlib.Path(__file__).parent / "cvrp_local_search.so"
    try:
        lib = ctypes.CDLL(c_cvrp_local_search_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(c_cvrp_local_search)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(c_cvrp_local_search_so_path)
    c_cvrp_local_search = lib.cvrp_local_search
    c_cvrp_local_search.argtypes = [
        ctypes.POINTER(ctypes.c_short), # tour
        ctypes.POINTER(ctypes.c_float), # coords
        ctypes.POINTER(ctypes.c_float), # demands
        ctypes.c_int,                   # nodes_num
        ctypes.c_int,                   # tour_length
        ctypes.c_int,                   # coords_scale
        ctypes.c_int,                   # demands_scale
        ctypes.c_int,                   # seed
    ]
    c_cvrp_local_search.restype = ctypes.POINTER(ctypes.c_int)
