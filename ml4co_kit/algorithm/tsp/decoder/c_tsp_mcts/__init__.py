import ctypes
import platform
import os
import pathlib


os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_mcts_decoder_path = pathlib.Path(__file__).parent
    c_mcts_decoder_so_path = pathlib.Path(__file__).parent / "tsp_mcts_decoder.so"
    try:
        lib = ctypes.CDLL(c_mcts_decoder_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(c_mcts_decoder_path)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(c_mcts_decoder_so_path)
    c_mcts_decoder = lib.mcts_decoder
    c_mcts_decoder.argtypes = [
        ctypes.POINTER(ctypes.c_float), # heatmap
        ctypes.POINTER(ctypes.c_float), # points
        ctypes.c_int,                   # nodes_num
        ctypes.c_int,                   # depth
        ctypes.c_float,                 # time_limit
        ctypes.c_int,                   # version_2opt [1/2]
        ctypes.c_int,                   # max_iterations_2opt                 
    ]
    c_mcts_decoder.restype = ctypes.POINTER(ctypes.c_int)
