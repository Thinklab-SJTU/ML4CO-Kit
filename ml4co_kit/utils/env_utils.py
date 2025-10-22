r"""
The utilities used to install the environment.
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
import platform
import importlib.util
import gurobipy as gp
from packaging import version


class EnvChecker(object):
    def __init__(self):
        # System
        self.system = platform.system()

        # Basic (torch)
        self.torch_support = self._check_package("torch")
        
        # GNN4CO (scipy, wandb, pytorch_ligntning, torch-X)
        self.scipy_support = self._check_package("scipy")
        self.torch_scatter_support = self._check_package("torch_scatter")
        self.torch_sparse_support = self._check_package("torch_sparse")
        self.torch_spline_conv_support = self._check_package("torch_spline_conv")
        self.torch_cluster_support = self._check_package("torch_cluster")
        self.wandb_support = self._check_package("wandb")
        self.pytorch_lightning_support = self._check_package("pytorch_lightning")
        
        # Gurobi
        try:
            env = gp.Env(empty=True)
            env.start()
            self.gurobi_support = True
        except:
            self.gurobi_support = False
        
        # Cuda
        self.cuda_support = None
        
    def _check_package(self, pkg: str) -> bool:
        return importlib.util.find_spec(pkg) is not None
    
    def check_gnn4co(self) -> bool:
        check_list = [
            self.torch_support,
            self.scipy_support,
            self.torch_scatter_support,
            self.torch_sparse_support,
            self.torch_spline_conv_support,
            self.torch_cluster_support,
            self.wandb_support,
            self.pytorch_lightning_support
        ]
        return all(check_list)
    
    def check_torch(self) -> bool:
        return self.torch_support
    
    def check_gurobi(self) -> bool:
        return self.gurobi_support

    def check_cuda(self) -> bool:
        if self.cuda_support is None:
            if self.check_torch():
                import torch
                self.cuda_support = torch.cuda.is_available()
            else:
                self.cuda_support = False
        return self.cuda_support
            
            
class EnvInstallHelper(object):
    def __init__(
        self, 
        pytorch_version: str = "2.1.0",
        use_cuda: bool = False,
        cuda_version: str = "121",
    ):
        self.pytorch_version = pytorch_version
        self.use_cuda = use_cuda
        self.cuda_version = cuda_version
    
    def install(self):
        # numpy & torch
        if version.parse(self.pytorch_version) < version.parse("2.4.0"):
            os.system(f"pip install 'numpy<2'")
        os.system(f"pip install torch=={self.pytorch_version}")
            
        # scipy
        os.system(f"pip install 'scipy>=1.10.1'")
        
        # torch-X (scatter, sparse, spline-conv, cluster)
        if self.use_cuda:
            torch_name = f"torch-{self.pytorch_version}+cu{self.cuda_version}"
        else:
            torch_name = f"torch-{self.pytorch_version}%2Bcpu"
        html_link = f"https://pytorch-geometric.com/whl/{torch_name}.html"
        os.system(f"pip install --no-index torch-scatter -f {html_link}")
        os.system(f"pip install --no-index torch-sparse -f {html_link}")
        os.system(f"pip install --no-index torch-spline-conv -f {html_link}")
        os.system(f"pip install --no-index torch-cluster -f {html_link}")
        
        # wandb
        os.system(f"pip install 'wandb>=0.20.0'")
        
        # pytorch-lightning
        os.system(f"pip install pytorch-lightning=={self.pytorch_version}")