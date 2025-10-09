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
numpy>=1.24.4
networkx>=2.8.8
tqdm>=4.66.3
pulp>=2.8.0, 
pandas>=2.0.0,
scipy>=1.10.1
aiohttp>=3.10.11
requests>=2.32.0
async_timeout>=4.0.3
pyvrp>=0.6.3
cython>=3.0.8
gurobipy>=11.0.3
scikit-learn>=1.3.0
matplotlib>=3.7.4
```

To ensure you have access to all functions, you need to install the environment related to ``pytorch_lightning``. We have provided an installation helper, and you can install it using the following code.

```python
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


## 📝 **ML4CO-Kit Development status**

We will present the development progress of ML4CO-Kit in the above 5 levels. 

**Graph: MCl & MCut & MIS & MVC; ✔: Supported; 📆: Planned for future versions (contributions welcomed!).**

### **Task (Level 1)**

| Task | Definition | Check Constraint | Evaluation | Render | Special R/O |
| ---- | :--------: | :--------------: | :--------: | :----: | :---------: |
|  Asymmetric TSP (ATSP)                        | ✔ | ✔ | ✔ | 📆 | ``tsplib`` |
|  Capacitated Vehicle Routing Problem (CVRP)   | ✔ | ✔ | ✔ | ✔  | ``vrplib`` |
|  Orienteering Problem (OP)                    | ✔ | ✔ | ✔ | 📆 |   |
|  Prize Collection TSP (PCTSP)                 | ✔ | ✔ | ✔ | 📆 |   |
|  Stochastic PCTSP (SPCTSP)                    | ✔ | ✔ | ✔ | 📆 |   |
|  Traveling Salesman Problem (TSP)             | ✔ | ✔ | ✔ | ✔  | ``tsplib`` |
|  Maximum Clique (MCl)                         | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
|  Maximum Cut (MCut)                           | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
|  Maximum Independent Set (MIS)                | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |
|  Minimum Vertex Cover (MVC)                   | ✔ | ✔ | ✔ | ✔  | ``gpickle``, ``adj_matrix``, ``networkx``, ``csr`` |

### **Generator (Level 2)**

| Task | Distribution | Brief Intro. | State |
| :--: | :----------: | ------------ | :---: |
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

### **Solver (Level 3)**

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

### **Optimizer (Level 4)**

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

### **Wrapper (Level 5)**

| Wrapper | TXT | Other R&W |
| :-----: | --- | :-------: |
| ATSPWrapper | "[dists] output [sol]" | ``tsplib`` |
| CVRPWrapper | "depots [depots] points [points] demands [demands] capacity [capacity] output [sol]" | ``vrplib`` |
| ORWrapper | "depots [depots] points [points] prizes [prizes] max_length [max_length] output [sol]" | |
| PCTSPWrapper | "depots [depots] points [points] penalties [penalties] prizes [prizes] required_prize [required_prize] output [sol]" | |
| SPCTSPWrapper | "depots [depots] points [points] penalties [penalties] expected_prizes [expected_prizes] actual_prizes [actual_prizes] required_prize [required_prize] output [sol]" | |
| TSPWrapper | "[points] output [sol]" | ``tsplib`` |
| (Graph)Wrapper | "[edge_index] label [sol]" | ``gpickle`` |
| (Graph)Wrapper [weighted]| "[edge_index] weights [weights] label [sol]" | ``gpickle`` |


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
