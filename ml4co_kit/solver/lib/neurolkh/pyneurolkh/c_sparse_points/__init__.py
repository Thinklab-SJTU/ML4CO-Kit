import os
import ctypes
import pathlib
import platform

os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    sparse_points_path = pathlib.Path(__file__).parent
    sparse_points_so_path = pathlib.Path(__file__).parent / "SparsePoints.so"
    try:
        lib = ctypes.CDLL(sparse_points_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(sparse_points_path)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(sparse_points_so_path)
    c_sparse_points = lib.SparsePoints
    c_sparse_points.argtypes = [
        ctypes.c_int, # nodes_num
        ctypes.POINTER(ctypes.c_float), # coords   
        ctypes.c_int, # sparse_factor              
        ctypes.c_float, # scale                             
    ]
    c_sparse_points.restype = ctypes.POINTER(ctypes.c_int)
