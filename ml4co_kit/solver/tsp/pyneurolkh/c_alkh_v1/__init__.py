import ctypes
import platform
import os
import pathlib

os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    alkh_path = pathlib.Path(__file__).parent
    alkh_so_path = pathlib.Path(__file__).parent / "ALKH.so"
    try:
        lib = ctypes.CDLL(alkh_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(alkh_path)
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(alkh_so_path)
    c_alkh_tool_v1 = lib.ALKH
    c_alkh_tool_v1.argtypes = [
        ctypes.c_int, # nodes_num
        ctypes.POINTER(ctypes.c_float), # coords
        ctypes.POINTER(ctypes.c_float), # penalty  
        ctypes.c_int, # candidates_num(in)              
        ctypes.c_int, # candidates_num(out)              
        ctypes.c_float, # scale               
        ctypes.c_float, # lr               
        ctypes.c_int, # initial period               
    ]
    c_alkh_tool_v1.restype = ctypes.POINTER(ctypes.c_double)
