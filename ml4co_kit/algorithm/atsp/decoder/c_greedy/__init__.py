import ctypes
import platform
import os
import pathlib


os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    c_atsp_greedy_decoder_path = pathlib.Path(__file__).parent
    c_atsp_greedy_decoder_so_path = pathlib.Path(__file__).parent / "atsp_greedy_decoder.so"
    try:
        lib = ctypes.CDLL(c_atsp_greedy_decoder_so_path)
    except:
        ori_dir = os.getcwd()
        os.chdir(c_atsp_greedy_decoder_path)
        os.system("gcc ./atsp_greedy_decoder.c -o atsp_greedy_decoder.so -fPIC -shared")
        os.chdir(ori_dir)
        lib = ctypes.CDLL(c_atsp_greedy_decoder_so_path)
    c_atsp_greedy_decoder = lib.nearest_neighbor

