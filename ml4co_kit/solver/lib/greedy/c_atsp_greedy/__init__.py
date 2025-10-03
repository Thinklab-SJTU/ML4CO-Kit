r"""
C++ Solver for ATSP Greedy
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

