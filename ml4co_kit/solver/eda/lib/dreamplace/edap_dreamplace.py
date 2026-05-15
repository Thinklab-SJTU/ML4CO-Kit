r"""
DreamPlace Solver for EDA Problems
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import sys
import time
import pathlib
import logging
from ml4co_kit.task.eda.edap import EDAPTask
from ml4co_kit.task.eda.c_edap_reader import ISPD2005Reader


def edap_dreamplace(task_data: EDAPTask, params):
    """Run DREAMPlace global placement using DreamPlaceParams."""

    # Import DreamPlace modules
    from dreamplace.Placer import place
    from dreamplace.Params import Params as DreamPlaceParams
    params: DreamPlaceParams

    # Initialize logging
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    logging.info(
        "EDAP task: benchmark_name=%s",
        getattr(task_data, "benchmark_name", None),
    )
    logging.info("parameters = %s" % (params,))

    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # Get learning rate
    lr = params.__dict__.get("global_place_stages")[0].get("learning_rate")

    # Call DreamPlace
    tt = time.time()
    place(params, lr)
    logging.info("placement takes %.3f seconds" % (time.time() - tt))

    # Get result path
    name = task_data.name
    result_dir: pathlib.Path = task_data.cache["ispd2005_result_dir"]
    result_path = result_dir / f"{name}/{name}.gp.pl"
    logging.info("Result is saved to %s" % (result_path.as_posix()))

    # Read the placement result
    reader = ISPD2005Reader()
    sol = reader.from_lg_pl(str(result_path))
    task_data.from_data(sol=sol, ref=False)