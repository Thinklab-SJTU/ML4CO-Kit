r"""
Pickle utilities with legacy module path remapping.
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


import pickle
from typing import Any, BinaryIO, Dict


# Legacy module paths used in older pickle files -> current module paths.
MODULE_REMAP: Dict[str, str] = {
    "ml4co_kit.task.routing.cvrp": "ml4co_kit.task.routing.vrp.cvrp",
    "ml4co_kit.task.routing.cvrpb": "ml4co_kit.task.routing.vrp.cvrpb",
    "ml4co_kit.task.routing.cvrpl": "ml4co_kit.task.routing.vrp.cvrpl",
    "ml4co_kit.task.routing.cvrptw": "ml4co_kit.task.routing.vrp.cvrptw",
    "ml4co_kit.task.routing.cvrpbl": "ml4co_kit.task.routing.vrp.cvrpbl",
    "ml4co_kit.task.routing.cvrpbtw": "ml4co_kit.task.routing.vrp.cvrpbtw",
    "ml4co_kit.task.routing.cvrpltw": "ml4co_kit.task.routing.vrp.cvrpltw",
    "ml4co_kit.task.routing.cvrpbltw": "ml4co_kit.task.routing.vrp.cvrpbltw",
    "ml4co_kit.task.routing.tsp": "ml4co_kit.task.routing.tsp.tsp",
    "ml4co_kit.task.routing.atsp": "ml4co_kit.task.routing.tsp.atsp",
    "ml4co_kit.task.routing.op": "ml4co_kit.task.routing.tsp.op",
    "ml4co_kit.task.routing.pctsp": "ml4co_kit.task.routing.tsp.pctsp",
    "ml4co_kit.task.routing.spctsp": "ml4co_kit.task.routing.tsp.spctsp",
}


# Legacy class names used in older pickle files -> current class names.
CLASS_NAME_REMAP: Dict[str, str] = {
    "DisntanceEvaluator": "DistanceEvaluator",
}


class LegacyUnpickler(pickle.Unpickler):
    """Unpickler that remaps legacy module paths to their current locations."""

    def find_class(self, module: str, name: str):
        module = MODULE_REMAP.get(module, module)
        name = CLASS_NAME_REMAP.get(name, name)
        return super().find_class(module, name)


def load_pickle(file: BinaryIO) -> Any:
    """Load a pickle file with legacy module path remapping."""
    return LegacyUnpickler(file).load()
