import ctypes
import platform
import os
import pathlib


os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_insertion_path = pathlib.Path(__file__).parent
    c_insertion_so_path = pathlib.Path(__file__).parent / "insertion.so"
    try:
        lib = ctypes.CDLL(c_insertion_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(c_insertion_path)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(c_insertion_so_path)
    c_insertion = lib.insertion
    c_insertion.argtypes = [
        ctypes.POINTER(ctypes.c_short), # order
        ctypes.POINTER(ctypes.c_float), # points
        ctypes.c_int,                   # nodes_num              
    ]
    c_insertion.restype = ctypes.POINTER(ctypes.c_int)
