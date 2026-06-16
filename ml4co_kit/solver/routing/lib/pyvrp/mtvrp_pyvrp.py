r"""
Solve MTVRP using PyVRP
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


from typing import Dict, Tuple, Callable
from ml4co_kit.task.routing.vrp.mtvrp import MTVRPTask
from .cvrp_pyvrp import cvrp_pyvrp
from .cvrpb_pyvrp import cvrpb_pyvrp
from .cvrpl_pyvrp import cvrpl_pyvrp
from .cvrptw_pyvrp import cvrptw_pyvrp
from .cvrpbl_pyvrp import cvrpbl_pyvrp
from .cvrpbtw_pyvrp import cvrpbtw_pyvrp
from .cvrpltw_pyvrp import cvrpltw_pyvrp
from .cvrpbltw_pyvrp import cvrpbltw_pyvrp


def mtvrp_pyvrp(
    task_data: MTVRPTask,
    time_limit: float = 1.0,
    scale: int = int(1e5),
    seed: int = 1234,
):
    # Get flags
    backhaul_flag = task_data.backhaul_flag
    max_route_length_flag = task_data.max_route_length_flag
    tw_flag = task_data.tw_flag

    # Function dict (B/MB, L, TW)
    function_dict: Dict[Tuple[bool, bool, bool], Callable] = {
        (False, False, False): cvrp_pyvrp,
        (False, False, True): cvrptw_pyvrp,
        (False, True, False): cvrpl_pyvrp,
        (False, True, True): cvrpltw_pyvrp,
        (True, False, False): cvrpb_pyvrp,
        (True, False, True): cvrpbtw_pyvrp,
        (True, True, False): cvrpbl_pyvrp,
        (True, True, True): cvrpbltw_pyvrp,
    }

    # Call the corresponding function
    solve_func = function_dict[(backhaul_flag, max_route_length_flag, tw_flag)]
    return solve_func(task_data, time_limit, scale, seed)