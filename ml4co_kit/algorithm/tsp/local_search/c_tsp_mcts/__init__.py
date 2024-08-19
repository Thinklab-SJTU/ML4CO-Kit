import ctypes
import platform
import os
import pathlib

os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_mcts_path = pathlib.Path(__file__).parent
    c_mcts_so_path = pathlib.Path(__file__).parent / "mcts.so"
    try:
        lib = ctypes.CDLL(c_mcts_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(c_mcts_path)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(c_mcts_so_path)
    c_mcts_local_search = lib.mcts_local_search
    c_mcts_local_search.argtypes = [
        ctypes.POINTER(ctypes.c_short), # tour
        ctypes.POINTER(ctypes.c_float), # heatmap
        ctypes.POINTER(ctypes.c_float), # points
        ctypes.c_int,                   # nodes_num
        ctypes.c_int,                   # depth
        ctypes.c_float,                 # time_limit
        ctypes.c_int,                   # version_2opt [1/2]
        ctypes.c_int,                   # continue_flag [1/2]
        ctypes.c_int,                   # max_iterations_2opt
    ]
    c_mcts_local_search.restype = ctypes.POINTER(ctypes.c_int)
