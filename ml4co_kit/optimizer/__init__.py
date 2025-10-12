r"""
Optimizer module for ML4CO-Kit.

This module provides various optimization algorithms for combinatorial optimization problems.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


# Check if torch is supported
from ml4co_kit.utils.env_utils import EnvChecker
env_checker = EnvChecker()
if env_checker.check_torch():
    from .two_opt import TwoOptOptimizer
    from .mcts import MCTSOptimizer
    from .rlsa import RLSAOptimizer


# Load other optimizers
from .base import OptimizerBase, OPTIMIZER_TYPE
from .cvrp_ls import CVRPLSOptimizer