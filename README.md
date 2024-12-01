<h1 align="center">
<img src="https://raw.githubusercontent.com/Thinklab-SJTU/ML4CO-Kit/main/docs/assets/ml4co-kit-logo.png" width="800">
</h1>

[![PyPi version](https://badgen.net/pypi/v/ml4co-kit/)](https://pypi.org/pypi/ml4co_kit/) [![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fml4co_kit%2Fjson)](https://pypi.python.org/pypi/ml4co-kit/) [![Downloads](https://static.pepy.tech/badge/ml4co-kit)](https://pepy.tech/project/ml4co-kit) [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/ML4CO-Kit.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/ML4CO-Kit/stargazers/)

Combinatorial Optimization (CO) is a mathematical optimization area that involves finding the best solution from a large set of discrete possibilities, often under constraints. Widely applied in routing, logistics, hardware design, and biology, CO addresses NP-hard problems critical to computer science and industrial engineering.

`ML4CO-Kit` aims to provide foundational support for machine learning practices on CO problems, including the follow aspects. 

* ``algorithm``: common post-processing algorithms.
* ``data``: common test datasets and our generated traning dataset.
* ``draw``: visualization of problems and solutions.
* ``evaluate``: evaluator for problems and solvers.
* ``generator``: data generation of various distributions.
* ``learning``: implemented base classes that facilitate method development for ML4CO.
* ``solver``: solvers' base classes and mainstream traditional solvers.
* ``utils``: general or commonly used functions and classes.


⭐ **Official Documentation**: https://ML4CO-Kit.readthedocs.io

⭐ **Source Code**: https://github.com/Thinklab-SJTU/ML4CO-Kit


## Development status

#### Basic

| Problem | Generator | Basic Solver (IO) | Traditional Solver |
| :-----: | :-------: | :---------------: | :----------------: |
|  ATSP   | ``sat``, ``hcp``, ``uniform`` | ``tsplib``, ``txt`` | ``LKH`` |
|  CVRP   | ``uniform``, ``gaussian`` | ``vrplib``, ``txt`` | ``LKH``, ``HGS``, ``PyVRP`` |
|  MCl    | ``er``, ``ba``, ``hk``, ``ws`` | ``gpickle``, ``txt``, ``networkx`` | ``Gurobi`` |
|  MCut   | ``er``, ``ba``, ``hk``, ``ws`` | ``gpickle``, ``txt``, ``networkx`` | ``Gurobi`` |
|  MIS    | ``er``, ``ba``, ``hk``, ``ws`` | ``gpickle``, ``txt``, ``networkx`` | ``Gurobi``, ``KaMIS`` |
|  MVC    | ``er``, ``ba``, ``hk``, ``ws`` | ``gpickle``, ``txt``, ``networkx`` | ``Gurobi`` |
|  TSP    | ``uniform``, ``gaussian``, ``cluster`` | ``tsplib``, ``txt`` | ``LKH``, ``Concorde``, ``GAX`` |

#### Extension

| Problem | Visualization | Algorithm | Test Dataset | Train Dataset |
| :-----: | :-----------: | :-------: | :----------: | :-----------: |
|  ATSP   | 📆 | 2 | 📆 | 📆 |
|  CVRP   | ✔  | 📆 | ``vrplib``, ``uniform`` | 📆 |
|  MCl    | ✔  | 📆 | 📆  | 📆 |
|  MCut   | ✔  | 📆 | 📆 | 📆 |
|  MIS    | ✔  | 📆 | 📆 | 📆 |
|  MVC    | ✔  | 📆 | 📆 | 📆 |
|  TSP    | ✔  | 4 | ``satlib``, ``uniform`` | ``uniform`` |

1~9: Number of supports; ✔: Supported; 📆: Planned for future versions (contributions welcomed!).

**We are still enriching the library and we welcome any contributions/ideas/suggestions from the community.**

**A comprehensive modular framework built upon this library that integrates core ML4CO technologies is coming soon.**

## Installation

You can install the stable release on PyPI:

```bash
$ pip install ml4co-kit
```

or get the latest version by running:

```bash
$ pip install -U https://github.com/Thinklab-SJTU/ML4CO-Kit/archive/master.zip # with --user for user install (no root)
```

The following packages are required and shall be automatically installed by ``pip``:

```
Python>=3.8
numpy>=1.24.4
networkx>=2.8.8
tqdm>=4.66.1
pulp>=2.8.0, 
pandas>=2.0.0,
scipy>=1.10.1
aiohttp>=3.9.3
requests>=2.31.0
async_timeout>=4.0.3
pyvrp>=0.6.3
cython>=3.0.8
gurobipy>=11.0.3
```

To ensure you have access to all functions, such as visualization, you'll need to install the following packages using `pip`:

```
matplotlib
pytorch_lightning
```