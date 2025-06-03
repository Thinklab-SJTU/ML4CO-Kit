import os
import ctypes
import shutil
import pathlib
import platform
import numpy as np
from numpy.ctypeslib import ndpointer


def recompile():
    kamis_path = pathlib.Path(__file__).parent

    if os.path.exists(kamis_path / "KaMIS/deploy/"):
        shutil.rmtree(kamis_path / "KaMIS/deploy/")
    if os.path.exists(kamis_path / "KaMIS/tmp_build/"):
        shutil.rmtree(kamis_path / "KaMIS/tmp_build/")
    shutil.copytree(
        kamis_path / "kamis-source/", kamis_path / "KaMIS/tmp_build/"
    )
    ori_dir = os.getcwd()
    os.chdir(kamis_path / "KaMIS/tmp_build/")
    os.system("bash cleanup.sh")
    os.system("bash compile_withcmake.sh")
    os.chdir(ori_dir)
    shutil.copytree(
        kamis_path / "KaMIS/tmp_build/deploy/",
        kamis_path / "KaMIS/deploy/",
    )
    shutil.rmtree(kamis_path / "KaMIS/tmp_build/")
        

os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_arw_path = pathlib.Path(__file__).parent
    
    # mis_two_improve
    c_mis_two_improve_lib_path = pathlib.Path(__file__).parent / "KaMIS/deploy/libmis_two_improve.so"
    try:
        c_mis_two_improve_lib = ctypes.CDLL(c_mis_two_improve_lib_path)
    except:
        recompile()
        c_mis_two_improve_lib = ctypes.CDLL(c_mis_two_improve_lib_path)
        
    c_mis_two_improve = c_mis_two_improve_lib.mis_two_improve
    c_mis_two_improve.argtypes = [
        ctypes.c_int,                                     # num_nodes
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # xadj
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # adjncy
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # initial_solution
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")   # output
    ]
    
    # mis_three_improve
    c_mis_three_improve_lib_path = pathlib.Path(__file__).parent / "KaMIS/deploy/libmis_three_improve.so"
    try:
        c_mis_three_improve_lib = ctypes.CDLL(c_mis_three_improve_lib_path)
    except:
        recompile()
        c_mis_three_improve_lib = ctypes.CDLL(c_mis_three_improve_lib_path)
        
    c_mis_three_improve = c_mis_three_improve_lib.mis_three_improve
    c_mis_three_improve.argtypes = [
        ctypes.c_int,                                     # num_nodes
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # xadj
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # adjncy
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # initial_solution
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")   # output
    ]

    # mis_ils
    c_mis_ils_lib_path = pathlib.Path(__file__).parent / "KaMIS/deploy/libmis_ils.so"
    try:
        c_mis_ils_lib = ctypes.CDLL(c_mis_ils_lib_path)
    except:
        recompile()
        c_mis_ils_lib = ctypes.CDLL(c_mis_ils_lib_path)
        
    c_mis_ils = c_mis_ils_lib.mis_ils
    c_mis_ils.argtypes = [
        ctypes.c_int,                                     # num_nodes
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # xadj
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # adjncy
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # initial_solution
        ctypes.c_int,                                     # iter
        ctypes.c_int,                                     # use_three_ls
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # output
    ]

    # mis_evo
    c_mis_evo_lib_path = pathlib.Path(__file__).parent / "KaMIS/deploy/libmis_evo.so"
    try:
        c_mis_evo_lib = ctypes.CDLL(c_mis_evo_lib_path)
    except:
        recompile()
        c_mis_evo_lib = ctypes.CDLL(c_mis_evo_lib_path)
        
    c_mis_evo = c_mis_evo_lib.mis_evo
    c_mis_evo.argtypes = [
        ctypes.c_int,                                     # num_nodes
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # xadj
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # adjncy
        ctypes.c_double,                                  # time_limit
        ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # output
    ]