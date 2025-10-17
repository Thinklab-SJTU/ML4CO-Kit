r"""
Utils Module.
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


r"""
Utils Functions and Modules.
"""

import importlib.util

# Env Utils
from .env_utils import EnvInstallHelper, EnvChecker

# File Utils
from .file_utils import (
    download, pull_file_from_huggingface, get_md5,
    compress_folder, extract_archive, check_file_path
)

# Time Utils
from .time_utils import Timer, tqdm_by_time

# Type Utils
if importlib.util.find_spec("torch") is not None:
    from .type_utils import to_numpy, to_tensor