r"""
DreamPlace Solver for EDA Problems
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
import sys
import time
import logging
from dreamplace.Params import Params
from dreamplace.Placer import place
from ml4co_kit.task.eda.edap import EDAPTask


def edap_dreamplace(task_data: EDAPTask):
    """
    Solve the EDA problem using DreamPlace solver.
    """
    # Set up logging
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)-7s] %(name)s - %(message)s',
        stream=sys.stdout
    )

    # Get parameters path
    params_path = "ml4co_kit/extension/dreamplace/source/test/ispd2005/adaptec1.json"

    # Load parameters
    params = Params()
    params.printWelcome()
    params.load(params_path)
    logging.info("parameters = %s" % (params))
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # Extract the learning rate value from the json file 
    # to assign it to the optimizer of the "torch_optimizer" package
    lr = params.__dict__.get('global_place_stages')[0].get('learning_rate')
    
    # Run placement
    tt = time.time()
    place(params, lr)
    logging.info("placement takes %.3f seconds" % (time.time() - tt))