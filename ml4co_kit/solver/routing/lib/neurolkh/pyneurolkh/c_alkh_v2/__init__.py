r"""
C++ Solver for ALKH v2
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import ctypes
import pathlib
import platform


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
    c_alkh_tool_v2 = lib.ALKH
    c_alkh_tool_v2.argtypes = [
        ctypes.c_int, # nodes_num
        ctypes.POINTER(ctypes.c_float), # coords
        ctypes.POINTER(ctypes.c_float), # penalty  
        ctypes.POINTER(ctypes.c_int32), # candidates  
        ctypes.c_int, # candidates_num(in)              
        ctypes.c_int, # candidates_num(out)              
        ctypes.c_float, # scale               
        ctypes.c_float, # lr               
        ctypes.c_int, # initial period               
    ]
    c_alkh_tool_v2.restype = ctypes.POINTER(ctypes.c_double)
