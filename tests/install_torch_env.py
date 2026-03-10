r"""
Install PyTorch Environment.
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
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)
from packaging import version
from ml4co_kit import EnvInstallHelper


if __name__ == "__main__":
    # Get pytorch version
    python_version = sys.version.split()[0]
    
    # Get pytorch version
    if version.parse(python_version) < version.parse("3.12"):
        pytorch_version = "2.1.0"
    elif version.parse(python_version) < version.parse("3.13"):
        pytorch_version = "2.4.0"
    else:
        pytorch_version = "2.7.0"
    
    # Install pytorch environment
    env_install_helper = EnvInstallHelper(pytorch_version=pytorch_version)
    env_install_helper.install()