r"""
MS Learning Module (Ascend-oriented backend).

Exports ``BaseEnv`` always; exports ``BaseModel`` / ``Trainer`` / helpers only
when ``check_learning(backend="mindspore")`` succeeds (``mindspore`` installed).

Interface mirrors ``ml4co_kit.learning.pytorch`` so higher-level code can switch
backends with minimal changes.
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


from ml4co_kit.utils.env_utils import EnvChecker

if EnvChecker().check_learning(backend="mindspore"):
    from .dataloader import MSDataset, MSDataLoader
    from .model import MSBaseModel
    from .train import MSCheckpoint, MSLogger, MSTrainer
