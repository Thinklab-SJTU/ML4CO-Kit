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

* **Task:** Level 1, the smallest processing unit, where each task represents a problem instance. At the task level, it mainly involves the definition of CO problems, evaluation of solutions (including constraint checking), and problem visualization, etc.
* **Generator:** Level 2, the generator creates task instances of a specific structure or distribution based on the set parameters.
* **Solver:** Level 3, a variety of solvers. Different solvers, based on their scope of application, can solve specific types of task instances and can be combined with optimizers to further improve the solution results.
* **Optimizer:** Level 4, to further optimize the initial solution obtained by the solver.
* **Wrapper:** Level 5, user-friendly wrappers, used for handling data reading and writing, task storage, as well as parallelized generation and solving.

Additionally, for higher-level ML4CO (see [ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)) services, we also provide learning base classes (see [ml4co_kit/learning](https://github.com/Thinklab-SJTU/ML4CO-Kit/tree/main/ml4co_kit/learning) ) based on the PyTorch-Lightning framework, including ``BaseEnv``, ``BaseModel``, ``Trainer``. The following figure illustrates the relationship between the ``ML4CO-Kit`` and ``ML4CO-Bench-101``.

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
```


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
