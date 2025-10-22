<h1 align="center">
<img src="https://raw.githubusercontent.com/Thinklab-SJTU/ML4CO-Kit/main/docs/assets/ml4co-kit-logo.png" width="800">
</h1>

[![PyPi version](https://badgen.net/pypi/v/ml4co-kit/)](https://pypi.org/pypi/ml4co_kit/) 
[![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fml4co_kit%2Fjson)](https://pypi.python.org/pypi/ml4co-kit/) 
[![Downloads](https://static.pepy.tech/badge/ml4co-kit)](https://pepy.tech/project/ml4co-kit) 
[![Documentation Status](https://readthedocs.org/projects/ml4co_kit/badge/?version=latest)](https://ml4co-kit.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/Thinklab-SJTU/ML4CO-Kit/branch/main/graph/badge.svg?token=5GGETAYIFL)](https://codecov.io/gh/Thinklab-SJTU/ML4CO-Kit)
[![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/ML4CO-Kit.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/ML4CO-Kit/stargazers/)

## 📚 Introductions

Combinatorial Optimization (CO) is a mathematical optimization area that involves finding the best solution from a large set of discrete possibilities, often under constraints. Widely applied in routing, logistics, hardware design, and biology, CO addresses NP-hard problems critical to computer science and industrial engineering.

`ML4CO-Kit` aims to provide foundational support for machine learning practices on CO problems.
We have designed the ``ML4CO-Kit`` into five levels: 

<img src="https://raw.githubusercontent.com/Thinklab-SJTU/ML4CO-Kit/main/docs/assets/organization.png" alt="Organization" width="600"/>

* **``Task``(Level 1):** the smallest processing unit, where each task represents a problem instance. At the task level, it mainly involves the definition of CO problems, evaluation of solutions (including constraint checking), and problem visualization, etc.
* **``Generator``(Level 2):** the generator creates task instances of a specific structure or distribution based on the set parameters.
* **``Solver``(Level 3):** a variety of solvers. Different solvers, based on their scope of application, can solve specific types of task instances and can be combined with optimizers to further improve the solution results.
* **``Optimizer``(Level 4):** to further optimize the initial solution obtained by the solver.
* **``Wrapper``(Level 5):** user-friendly wrappers, used for handling data reading and writing, task storage, as well as parallelized generation and solving.

Additionally, for higher-level ML4CO (see [ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)) services, we also provide learning base classes (see ``ml4co_kit/learning``) based on the PyTorch-Lightning framework, including ``BaseEnv``, ``BaseModel``, ``Trainer``. The following figure illustrates the relationship between the ``ML4CO-Kit`` and ``ML4CO-Bench-101``.

<img src="https://raw.githubusercontent.com/Thinklab-SJTU/ML4CO-Kit/main/docs/assets/relation.png" alt="Relation" width="400"/>

**We are still enriching the library and we welcome any contributions/ideas/suggestions from the community.**

⭐ **Official Documentation**: https://ml4co-kit.readthedocs.io/en/latest/

⭐ **Source Code**: https://github.com/Thinklab-SJTU/ML4CO-Kit


## 🚀 Installation

You can install the stable release on PyPI:

```bash
$ pip install ml4co-kit
```

<img src="https://raw.githubusercontent.com/Thinklab-SJTU/ML4CO-Kit/main/docs/assets/pip.png" alt="pip" width="300"/>

or get the latest version by running:

```bash
$ pip install -U https://github.com/Thinklab-SJTU/ML4CO-Kit/archive/master.zip # with --user for user install (no root)
```

The following packages are required and shall be automatically installed by ``pip``:

```
Python>=3.8
numpy>=1.24.3
networkx>=2.8.8
tqdm>=4.66.3
cython>=3.0.8
pulp>=2.8.0, 
scipy>=1.10.1
aiohttp>=3.10.11
requests>=2.32.0
matplotlib>=3.7.0
async_timeout>=4.0.3
pyvrp>=0.6.3
gurobipy>=11.0.3
scikit-learn>=1.3.0
ortools>=9.12.4544
huggingface_hub>=0.32.0
setuptools>=75.0.0
PySCIPOpt>=5.6.0
```

To ensure you have access to all functions, you need to install the environment related to ``pytorch_lightning``. We have provided an installation helper, and you can install it using the following code.

```python
import sys
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
```

⚠️ **2025-10-14:** While testing the NVIDIA GeForce RTX 50-series GPUs, we have encountered the following error. To fix this issue, we recommend that you upgrade your driver to version ``12.8`` or later and download the corresponding PyTorch build from the official PyTorch website.

```bash
XXX with CUDA capability sm_120 is not compatible with the current PyTorch installation. 
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

``` python
import os

# download torch==2.8.0+cu128 from pytorch.org
os.system(f"pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128")

# download torch-X (scatter, sparse, spline-conv, cluster)
html_link = f"https://pytorch-geometric.com/whl/torch-2.8.0+cu128.html"
os.system(f"pip install --no-index torch-scatter -f {html_link}")
os.system(f"pip install --no-index torch-sparse -f {html_link}")
os.system(f"pip install --no-index torch-spline-conv -f {html_link}")
os.system(f"pip install --no-index torch-cluster -f {html_link}")

# wandb
os.system(f"pip install wandb>=0.20.0")

# pytorch-lightning
os.system(f"pip install pytorch-lightning==2.5.3")
```

After the environment is installed, run the following command to confirm that the PyTorch build supports ``sm_120``.

```python
>>> import torch
>>> print(torch.cuda.get_arch_list())
['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

⚠️ **2025-10-21:** We find that on macOS, the ``gurobipy`` package does not support ``Python 3.8`` or earlier. Therefore, please upgrade your Python to at least 3.9.


## 📝 **ML4CO-Kit Development status**

We will present the development progress of ML4CO-Kit in the above 5 levels. 

**Graph: MCl & MCut & MIS & MVC; Portfolio: MaxRetPO & MinVarPO & MOPO**

**✔: Supported; 📆: Planned for future versions (contributions welcomed!).**

<details>
<summary>Task (Level 1)</summary>

| Task | Definition | Check Constraint | Evaluation | Render | Special R/O |
| ---- | :--------: | :--------------: | :--------: | :----: | :---------: |
| **Routing Tasks** |
|  Asymmetric TSP (ATSP)                              | ✔ | ✔ | ✔ | 📆 | ``tsplib`` |
|  Capacitated Vehicle Routing Problem (CVRP)         | ✔ | ✔ | ✔ | ✔  | ``vrplib`` |
|  Orienteering Problem (OP)                          | ✔ | ✔ | ✔ | 📆 |   |
|  Prize Collection TSP (PCTSP)                       | ✔ | ✔ | ✔ | 📆 |   |
|  Stochastic PCTSP (SPCTSP)                          | ✔ | ✔ | ✔ | 📆 |   |
|  Traveling Salesman Problem (TSP)                   | ✔ | ✔ | ✔ | ✔  | ``tsplib`` |
| **Graph Tasks** |
|  Maximum Clique (MCl)                               | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
|  Maximum Cut (MCut)                                 | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
|  Maximum Independent Set (MIS)                      | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
|  Minimum Vertex Cover (MVC)                         | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
| **Portfolio Tasks** |
|  Maximum Return Portfolio Optimization (MaxRetPO)   | ✔ | ✔ | ✔ | 📆  |  |
|  Minimum Variance Portfolio Optimization (MinVarPO) | ✔ | ✔ | ✔ | 📆  |  |
|  Multi-Objective Portfolio Optimization (MOPO)      | ✔ | ✔ | ✔ | 📆  |  |
</details>

---

<details>
<summary>Generator (Level 2)</summary>

| Task | Distribution | Brief Intro. | State |
| :--: | :----------: | ------------ | :---: |
| **Routing Tasks** |
| ATSP    | Uniform | Random distance matrix with triangle inequality | ✔ |
|         | SAT | SAT problem transformed to ATSP | ✔ |
|         | HCP | Hamiltonian Cycle Problem transformed to ATSP | ✔ |
| CVRP    | Uniform | Random coordinates with uniform distribution | ✔ |
|         | Gaussian | Random coordinates with Gaussian distribution | ✔ |
| OP      | Uniform | Random prizes with uniform distribution | ✔ |
|         | Constant | All prizes are constant | ✔ |
|         | Distance | Prizes based on distance from depot | ✔ |
| PCTSP   | Uniform | Random prizes with uniform distribution | ✔ |
| SPCTSP  | Uniform | Random prizes with uniform distribution | ✔ |
| TSP     | Uniform | Random coordinates with uniform distribution | ✔ |
|         | Gaussian | Random coordinates with Gaussian distribution | ✔ |
|         | Cluster | Coordinates clustered around random centers | ✔ |
| **Graph Tasks** |
| (Graph) | ER (structure) | Erdos-Renyi random graph | ✔ |
|         | BA (structure) | Barabasi-Albert scale-free graph | ✔ |
|         | HK (structure) | Holme-Kim small-world graph | ✔ |
|         | WS (structure) | Watts-Strogatz small-world graph | ✔ |
|         | RB (structure) | RB-Model graph | ✔ |
|         | Uniform (weighted) | Weights with Uniform distribution | ✔ |
|         | Gaussian (weighted) | Weights with Gaussian distribution | ✔ |
|         | Poisson (weighted) | Weights with Poisson distribution | ✔ |
|         | Exponential (weighted) | Weights with Exponential distribution | ✔ |
|         | Lognormal (weighted) | Weights with Lognormal distribution | ✔ |
|         | Powerlaw (weighted) | Weights with Powerlaw distribution | ✔ |
|         | Binomial (weighted) | Weights with Binomial distribution | ✔ |
| **Portfolio Tasks** |
| (Portfolio) | GBM | Geometric Brownian Motion model | ✔ |
|          | Factor | Factor model with k factors and idiosyncratic noise | ✔ |
|          | VAR(1) | Vector Autoregressive model of order 1 | ✔ |
|          | MVT | Multivariate T distribution model | ✔ |
|          | GRACH | GARCH model for volatility clustering | ✔ |
|          | Jump | Merton Jump-Diffusion model | ✔ |
|          | Regime | Regime-Switching model with multiple states | ✔ |

</details>

---

<details>
<summary>Solver (Level 3)</summary>

| Solver | Support Task | Language | Source | Ref. / Implementation | State | 
| :----: | :----------: |  ------- | :----: | :-------: | :---: |
| BeamSolver       | MCl   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MIS   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| ConcordeSolver   | TSP   | C/C++  | [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) | [PyConcorde](https://github.com/jvkersch/pyconcorde)  | ✔ |
| GAEAXSolver      | TSP   | C/C++  | [GA-EAX](https://github.com/nagata-yuichi/GA-EAX) | [GA-EAX](https://github.com/nagata-yuichi/GA-EAX) | ✔ |
| GpDegreeSolver   | MCl   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MIS   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MVC   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| GreedySolver     | ATSP  | C/C++  | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | CVRP  | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | TSP   | Cython | [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO/tree/main/difusco/utils/cython_merge) | [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO/tree/main/difusco/utils/cython_merge) | ✔ |
|                  | MCl   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MCut  | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MIS   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MVC   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| GurobiSolver     | ATSP  | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
|                  | CVRP  | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
|                  | OP    | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
|                  | TSP   | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
|                  | MCl   | C/C++  | [Gurobi](https://www.gurobi.com/) | [DIffUCO](https://github.com/ml-jku/DIffUCO) | ✔ |
|                  | MCut  | C/C++  | [Gurobi](https://www.gurobi.com/) | [DIffUCO](https://github.com/ml-jku/DIffUCO) | ✔ |
|                  | MIS   | C/C++  | [Gurobi](https://www.gurobi.com/) | [DIffUCO](https://github.com/ml-jku/DIffUCO) | ✔ |
|                  | MVC   | C/C++  | [Gurobi](https://www.gurobi.com/) | [DIffUCO](https://github.com/ml-jku/DIffUCO) | ✔ |
|                  | MaxRetPO | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MinVarPO | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MOPO  | C/C++  | [Gurobi](https://www.gurobi.com/) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| HGSSolver        | CVRP  | C/C++  | [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) | [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) | ✔ |
| ILSSolver        | PCTSP | Python | [PCTSP](https://github.com/jordanamecler/PCTSP) | [PCTSP](https://github.com/jordanamecler/PCTSP) | ✔ |
|                  | SPCTSP| Python | [Attention](https://github.com/wouterkool/attention-learn-to-route) | [Attention](https://github.com/wouterkool/attention-learn-to-route) | ✔ |
| InsertionSolver  | TSP   | Python | [GLOP](https://github.com/henry-yeh/GLOP) | [GLOP](https://github.com/henry-yeh/GLOP) | ✔ |
| KaMISSolver      | MIS   | Python | [KaMIS](https://github.com/KarlsruheMIS/KaMIS) | [MIS-Bench](https://github.com/MaxiBoether/mis-benchmark-framework) | ✔ |
| LcDegreeSolver   | MCl   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MCut  | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MIS   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MVC   | Python | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| LKHSolver        | TSP   | C/C++  | [LKH](http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
|                  | ATSP  | C/C++  | [LKH](http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
|                  | CVRP  | C/C++  | [LKH](http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)  | ✔ |
| MCTSSolver       | TSP   | Python | [Att-GCRN](https://github.com/Spider-scnu/TSP) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| NeuroLKHSolver   | TSP   | Python | [NeuroLKH](https://github.com/liangxinedu/NeuroLKH) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| ORSolver         | ATSP  | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | OP    | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | PCTSP | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | TSP   | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MCl   | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MIS   | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MVC   | C/C++  | [OR-Tools](https://developers.google.cn/optimization/introduction) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| RLSASolver       | MCl   | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MCut  | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MIS   | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MVC   | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| SCIPSolver       | MaxRetPO | C/C++  | [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MinVarPO | C/C++  | [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                  | MOPO  | C/C++  | [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
</details>

---

<details>
<summary>Optimizer (Level 4)</summary>

| Optimizer | Support Task | Language | Source | Reference | State | 
| :-------: | :----------: |  ------- | :----: | :-------: | :---: |
| CVRPLSOptimizer     | CVRP   | C/C++  | [HGS-CVRP](https://github.com/vidalt/HGS-CVRP) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| MCTSOptimizer       | TSP    | C/C++  | [Att-GCRN](https://github.com/Spider-scnu/TSP) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| RLSAOptimizer       | MCl    | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                     | MCut   | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                     | MIS    | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                     | MVC    | Python | [RLSA](https://arxiv.org/abs/2502.00277) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
| TwoOptOptimizer     | ATSP   | C/C++  | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |
|                     | TSP    | Python | [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO/blob/main/difusco/utils/tsp_utils.py) | [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit) | ✔ |

</details>

---

<details>
<summary>Wrapper (Level 5)</summary>

| Wrapper | TXT | Other R&W |
| :-----: | --- | :-------: |
| **Routing Tasks** |
| ATSPWrapper               | "[dists] output [sol]" | ``tsplib`` |
| CVRPWrapper               | "depots [depots] points [points] demands [demands] capacity [capacity] output [sol]" | ``vrplib`` |
| ORWrapper                 | "depots [depots] points [points] prizes [prizes] max_length [max_length] output [sol]" | |
| PCTSPWrapper              | "depots [depots] points [points] penalties [penalties] prizes [prizes] required_prize [required_prize] output [sol]" | |
| SPCTSPWrapper             | "depots [depots] points [points] penalties [penalties] expected_prizes [expected_prizes] actual_prizes [actual_prizes] required_prize [required_prize] output [sol]" | |
| TSPWrapper                | "[points] output [sol]" | ``tsplib`` |
| **Graph Tasks** |
| (Graph)Wrapper            | "[edge_index] label [sol]" | ``gpickle`` |
| (Graph)Wrapper [weighted] | "[edge_index] weights [weights] label [sol]" | ``gpickle`` |
| **Portfolio Tasks** |
| MaxRetPOWrapper           | "[returns] cov [cov] max_var [max_var] output [sol]" | |
| MinVarPOWrapper           | "[returns] cov [cov] required_returns [required_returns] output [sol]" | |
| MOPOWrapper               | "[returns] cov [cov] var_factor [var_factor] output [sol]" | |

</details>


## 🔎 **How to use ML4CO-Kit**

<details>
<summary>Case-01: How to use ML4CO-Kit to generate a dataset</summary>

```python
# We take the TSP as an example

# Import the required classes.
>>> import numpy as np                  # Numpy
>>> from ml4co_kit import TSPWrapper    # The wrapper for TSP, used to manage data and parallel generation.
>>> from ml4co_kit import TSPGenerator  # The generator for TSP, used to generate a single instance.
>>> from ml4co_kit import TSP_TYPE      # The distribution types supported by the generator.
>>> from ml4co_kit import LKHSolver     # We choose LKHSolver to solve TSP instances

# Check which distributions are supported by the TSP types.
>>> for type in TSP_TYPE:
...     print(type)
TSP_TYPE.UNIFORM
TSP_TYPE.GAUSSIAN
TSP_TYPE.CLUSTER

# Set the generator parameters according to the requirements.
>>> tsp_generator = TSPGenerator(
...     distribution_type=TSP_TYPE.GAUSSIAN,   # Generate a TSP instance with a Gaussian distribution
...     precision=np.float32,                  # Floating-point precision: 32-bit
...     nodes_num=50,                          # Number of nodes in TSP instance
...     gaussian_mean_x=0,                     # Mean of Gaussian for x coordinate
...     gaussian_mean_y=0,                     # Mean of Gaussian for y coordinate
...     gaussian_std=1,                        # Standard deviation of Gaussian
... )

# Set the LKH parameters.
>>> tsp_solver = LKHSolver(
...     lkh_scale=1e6,        # Scaling factor to convert floating-point numbers to integers
...     lkh_max_trials=500,   # Maximum number of trials for the LKH algorithm
...     lkh_path="LKH",       # Path to the LKH executable
...     lkh_runs=1,           # Number of runs for the LKH algorithm
...     lkh_seed=1234,        # Random seed for the LKH algorithm
...     lkh_special=False,    # When set to True, disables 2-opt and 3-opt heuristics
... )

# Create the TSP wrapper
>>> tsp_wrapper = TSPWrapper(precision=np.float32)

# Use ``generate_w_to_txt`` to generate a dataset of TSP.
>>> tsp_wrapper.generate_w_to_txt(
...     file_path="tsp_gaussian_16ins.txt",  # Path to the output file where the generated TSP instances will be saved
...     generator=tsp_generator,             # The TSP instance generator to use
...     solver=tsp_solver,                   # The TSP solver to use
...     num_samples=16,                      # Number of TSP instances to generate
...     num_threads=4,                       # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
...     batch_size=1,                        # Batch size for parallel processing; cannot both be non-1 with num_threads
...     write_per_iters=1,                   # Number of sub-generation steps after which data will be written to the file
...     write_mode="a",                      # Write mode for the output file ("a" for append)
...     show_time=True,                      # Whether to display the time taken for the generation process
... )
Generating TSP: 100%|██████████| 4/4 [00:00<00:00, 12.79it/s]
```

</details>

---

<details>
<summary>Case-02: How to use ML4CO-Kit to load problems and solve them</summary>

```python
# We take the MIS as an example

# Import the required classes.
>>> import numpy as np                  # Numpy
>>> from ml4co_kit import MISWrapper    # The wrapper for MIS, used to manage data and parallel solving.
>>> from ml4co_kit import KaMISSolver   # We choose KaMISSolver to solve MIS instances

# Set the KaMIS parameters.
>>> mis_solver = KaMISSolver(
...     kamis_time_limit=10.0,          # The maximum solution time for a single problem
...     kamis_weighted_scale=1e5,       # Weight scaling factor, used when nodes have weights.
... )

# Create the MIS wrapper
>>> mis_wrapper = MISWrapper(precision=np.float32)

# Load the problems to be solved.
# You can use the corresponding loading function based on the file type, 
# such as ``from_txt`` for txt file and ``from_pickle`` for pickle file.
>>> mis_wrapper.from_txt(
...     file_path="test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.txt",
...     ref=True,          # TXT file contains labels. Set ``ref=True`` to set them as reference.
...     overwrite=True,    # Whether to overwrite the data. If not, only update according to the file data.
...     show_time=True     # Whether to display the time taken for the loading process
... )
Loading data from test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.txt: 4it [00:00, 75.41it/s]

# Use ``solve`` to call the KaMISSolver to perform the solution.
>>> mis_wrapper.solve(
...     solver=mis_solver,                   # The solver to use
...     num_threads=2,                       # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
...     batch_size=1,                        # Batch size for parallel processing; cannot both be non-1 with num_threads
...     show_time=True,                      # Whether to display the time taken for the generation process
... )
Solving MIS Using kamis: 100%|██████████| 2/2 [00:21<00:00, 10.97s/it]
Using Time: 21.947036743164062

# Use ``evaluate_w_gap`` to obtain the evaluation results.
# Evaluation Results: average solution value, average reference value, gap (%), gap std.
>>> eval_result = mis_wrapper.evaluate_w_gap()
>>> print(eval_result)
(14.827162742614746, 15.18349838256836, 2.5054726600646973, 2.5342845916748047)
```

</details>

---

<details>
<summary>Case-03: How to use ML4CO-Kit to visualize the COPs </summary>

```python
# We take the CVRP as an example

# Import the required classes.
>>> import numpy as np                  # Numpy
>>> from ml4co_kit import CVRPTask      # CVRP Task. 
>>> from ml4co_kit import CVRPWrapper   # The wrapper for CVRP, used to manage data.

# Case-1: multiple task data are saved in ``txt``, ``pickle``, etc. single task data is saved in pickle.
>>> cvrp_wrapper = CVRPWrapper()
>>> cvrp_wrapper.from_pickle("test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.pkl")
>>> cvrp_task = cvrp_wrapper.task_list[0]
>>> print(cvrp_task)
CVRPTask(2fb389cdafdb4e79a94572f01edf0b95)

# Case-2: single task data is saved in pickle.
>>> cvrp_task = CVRPTask()
>>> cvrp_task.from_pickle("test_dataset/cvrp/task/cvrp50_uniform_task.pkl")
>>> print(cvrp_task)
CVRPTask(2fb389cdafdb4e79a94572f01edf0b95)

# The loaded solution is usually a reference solution. 
# When drawing the image, it is the ``sol`` that is being drawn. 
# Therefore, it is necessary to assign ``ref_sol`` to ``sol``.
>>> cvrp_task.sol = cvrp_task.ref_sol

# Using ``render`` to get the visualization
>>> cvrp_task.render(
...     save_path="./docs/assets/cvrp_solution.png",  # Path to save the rendered image
...     with_sol=True,                                # Whether to draw the solution tour
...     figsize=(10, 10),                             # Size of the image (width and height)
...     node_color="darkblue",                        # Color of the nodes
...     edge_color="darkblue",                        # Color of the edges
...     node_size=50                                  # Size of the nodes
... )
```
<img src="https://raw.githubusercontent.com/Thinklab-SJTU/ML4CO-Kit/main/docs/assets/cvrp_solution.png" alt="pip" width="300"/>

</details>

---

<details>
<summary>Case-04: A simple ML4CO example </summary>

```python
# We take the MCut as an example

# Import the required classes.
>>> import numpy as np                   # Numpy
>>> from ml4co_kit import MCutWrapper    # The wrapper for MCutWrapper, used to manage data.
>>> from ml4co_kit import GreedySolver   # GreedySolver, based on GNN4CO.
>>> from ml4co_kit import RLSAOptimizer  # Using RLSA to perform local search.
>>> from ml4co_kit.extension.gnn4co import GNN4COModel, GNN4COEnv, GNNEncoder

# Set the GNN4COModel parameters. ``weight_path``: Pretrain weight path. 
# If it is not available locally, it will be automatically downloaded from Hugging Face.
>>> gnn4mcut_model = GNN4COModel(
...     env=GNN4COEnv(
...         task="MCut",              # Task name: MCut.                                 
...         mode="solve",             # Mode: solving mode.
...         sparse_factor=1,          # Sparse factor: Controls the sparsity of the graph.
...         device="cuda"             # Device: 'cuda' or 'cpu'
...     ),
...     encoder=GNNEncoder(
...         task="MCut",              # Task name: MCut.
...         sparse=True,              # Graph data should set ``sparse`` to True.
...         block_layers=[2,4,4,2]    # Block layers: the number of layers in each block of the encoder.
...     ),
...     weight_path="weights/gnn4co_mcut_ba-large_sparse.pt"   
... )
gnn4co/gnn4co_mcut_ba-large_sparse.pt: 100% ███████████████ 19.6M/19.6M [00:03<00:00, 6.18MB/s]

# Set the RLSAOptimizer parameters.
>>> mcut_optimizer = RLSAOptimizer(
...     rlsa_kth_dim="both",          # Which dimension to consider for the k-th value calculation.
...     rlsa_tau=0.01,                # The temperature parameter in the Simulated Annealing process.
...     rlsa_d=2,                     # Control the step size of each update.
...     rlsa_k=1000,                  # The number of samples used in the optimization process.
...     rlsa_t=1000,                  # The number of iterations in the optimization process.
...     rlsa_device="cuda",           # Device: 'cuda' or 'cpu'.
...     rlsa_seed=1234                # The random seed for reproducibility.
... )

# Set the GreedySolver parameters.
>>> mcut_solver_wo_opt = GreedySolver(
...     model=gnn4mcut_model,         # GNN4CO model for MCut
...     device="cuda",                # Device: 'cuda' or 'cpu'.
...     optimizer=None                # The optimizer to perform local search.
... )
>>> mcut_solver_w_opt = GreedySolver(
...     model=gnn4mcut_model,         # GNN4CO model for MCut
...     device="cuda",                # Device: 'cuda' or 'cpu'.
...     optimizer=mcut_optimizer      # The optimizer to perform local search.
... )

# Create the MCut wrapper
>>> mcut_wrapper = MCutWrapper(precision=np.float32)

# Load the problems to be solved.
# You can use the corresponding loading function based on the file type, 
# such as ``from_txt`` for txt file and ``from_pickle`` for pickle file.
>>> mcut_wrapper.from_txt(
...     file_path="test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.txt",
...     ref=True,          # TXT file contains labels. Set ``ref=True`` to set them as reference.
...     overwrite=True,    # Whether to overwrite the data. If not, only update according to the file data.
...     show_time=True     # Whether to display the time taken for the loading process
... )
Loading data from test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.txt: 4it [00:00, 16.35it/s]

# Using ``solve`` to get the solution (without optimizer)
>>> mcut_wrapper.solve(
...     solver=mcut_solver_wo_opt,    # The solver to use
...     num_threads=1,                # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
...     batch_size=1,                 # Batch size for parallel processing; cannot both be non-1 with num_threads
...     show_time=True,               # Whether to display the time taken for the generation process
... )
Solving MCut Using greedy: 100%|██████████| 4/4 [00:00<00:00, 12.34it/s]
Using Time: 0.3261079788208008

# Use ``evaluate_w_gap`` to obtain the evaluation results.
# Evaluation Results: average solution value, average reference value, gap (%), gap std.
>>> eval_result = mcut_wrapper.evaluate_w_gap()
>>> print(eval_result)
(2647.25, 2726.5, 2.838811523236064, 0.7528157058230817)

# Using ``solve`` to get the solution (with optimizer)
>>> mcut_wrapper.solve(
...     solver=mcut_solver_w_opt,     # The solver to use
...     num_threads=1,                # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
...     batch_size=1,                 # Batch size for parallel processing; cannot both be non-1 with num_threads
...     show_time=True,               # Whether to display the time taken for the generation process
... )
Solving MCut Using greedy: 100%|██████████| 4/4 [00:02<00:00,  1.46it/s]
Using Time: 2.738525867462158

# Use ``evaluate_w_gap`` to obtain the evaluation results.
# Evaluation Results: average solution value, average reference value, gap (%), gap std.
>>> eval_result = mcut_wrapper.evaluate_w_gap()
>>> print(eval_result)
(2693.0, 2726.5, 1.2373146256952277, 0.29320238806274546)
```

</details>

## 📈 **Our Systematic Benchmark Works**

We are systematically building a foundational framework for ML4CO with a collection of resources that complement each other in a cohesive manner.

* [Awesome-ML4CO](https://github.com/Thinklab-SJTU/awesome-ml4co), a curated collection of literature in the ML4CO field, organized to support researchers in accessing both foundational and recent developments.

* [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit), a general-purpose toolkit that provides implementations of common algorithms used in ML4CO, along with basic training frameworks, traditional solvers and data generation tools. It aims to simplify the implementation of key techniques and offer a solid base for developing machine learning models for COPs.

* [ML4TSPBench](https://github.com/Thinklab-SJTU/ML4TSPBench): a benchmark focusing on exploring the TSP for representativeness. It advances a unified modular streamline incorporating existing tens of technologies in both learning and search for transparent ablation, aiming to reassess the role of learning and to discern which parts of existing techniques are genuinely beneficial and which are not. It offers a deep dive into various methodology designs, enabling comparisons and the development of specialized algorithms.

* [ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101): a benchmark that categorizes neural combinatorial optimization (NCO) solvers by solving paradigms, model designs, and learning strategies. It evaluates applicability and generalization of different NCO approaches across a broad range of combinatorial optimization problems to uncover universal insights that can be transferred across various domains of ML4CO.

* [PredictiveCO-Benchmark](https://github.com/Thinklab-SJTU/PredictiveCO-Benchmark): a benchmark for decision-focused learning (DFL) approaches on predictive combinatorial optimization problems.

## ✨ Citation
If you find our code helpful in your research, please cite
```
@inproceedings{
    ma2025mlcobench,
    title={ML4CO-Bench-101: Benchmark Machine Learning for Classic Combinatorial Problems on Graphs},
    author={Jiale Ma and Wenzheng Pan and Yang Li and Junchi Yan},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2025},
    url={https://openreview.net/forum?id=ye4ntB1Kzi}
}

@inproceedings{li2025unify,
  title={Unify ml4tsp: Drawing methodological principles for tsp and beyond from streamlined design space of learning and search},
  author={Li, Yang and Ma, Jiale and Pan, Wenzheng and Wang, Runzhong and Geng, Haoyu and Yang, Nianzu and Yan, Junchi},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
