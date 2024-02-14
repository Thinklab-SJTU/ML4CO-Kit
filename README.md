## Data4CO

[![PyPi version](https://badgen.net/pypi/v/data4co/)](https://pypi.org/pypi/data4co/)
[![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdata4co%2Fjson)](https://pypi.python.org/pypi/data4co/)
[![Downloads](https://static.pepy.tech/badge/data4co)](https://pepy.tech/project/data4co)
[![GitHub stars](https://img.shields.io/github/stars/heatingma/Data4CO.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/heatingma/Data4CO/stargazers/) 

A data generator tool for Combinatorial Optimization (CO) problems, enabling customizable, diverse, and scalable datasets for benchmarking optimization algorithms.

### Current support

**data**
|Problem|First|Impl.|Second|Impl.|Third|Impl.|
| :---: |:--:|:---:|:---:|:---:| :--: |:---:|
|  TSP  |tsplib| ✔ | LKH | ✔ | Concorde| ✔ |
|  MIS  |satlib| ✔ | KaMIS | 📆 | -- | -- |

**evaluator**
|Problem|First|Impl.|Second|Impl.|
| :---: |:--:|:---:|:---:|:---:|
|  TSP  |tsplib| 📆 | uniform | 📆 |
|  MIS  |satlib| 📆 | ER | 📆 |

**generator**
|Problem| Type1 |Impl.| Type2 |Impl.| Type3 |Impl.| Type4 |Impl.|
| :---: | :---: |:---:| :---: |:---:| :---: |:---:| :---: |:---:|
|  TSP  | uniform | ✔ | gaussian | ✔ | cluster | 📆 | -- | -- |
|  MIS  | ER | ✔ | BA | ✔ | HK | ✔ | WS | ✔ |

**solver**
|Problem|Base|Impl.|First|Impl.|Second|Impl.|
| :---: |:--:|:---:|:---:|:---:| :--: |:---:|
|  TSP  |TSPSolver| ✔ | LKH | ✔ | Concorde | ✔ |
|  MIS  | MISSolver | ✔ |KaMIS | ✔ | Gurobi| ✔ |

✔: Supported; 📆: Planned for future versions (contributions welcomed!).

### How to Install

**Github**
Clone with the url https://github.com/heatingma/Data4CO.git , and the following packages are required, and shall be automatically installed by ``pip``:
```
Python >= 3.8
numpy>=1.24.4
networkx==2.8.8
lkh>=1.1.1
tsplib95==0.7.1
tqdm>=4.66.1
pulp>=2.8.0, 
pandas>=2.0.0,
scipy>=1.10.1
```

**PyPI**
It is very convenient to directly use the following commands
```
pip install data4co
```

### How to Use Solver (TSPLKHSolver as example)

```python
from data4co.solver import TSPLKHSolver

tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=500)
tsp_lkh_solver.from_txt("path/to/read/file.txt")
tsp_lkh_solver.solve()
tsp_lkh_solver.evaluate()
tsp_lkh_solver.to_txt("path/to/write/file.txt")
```

### How to Use Generator (TSPDataGenerator as example)

```python
from data4co import TSPDataGenerator

tsp_data_lkh = TSPDataGenerator(
    num_threads=8,
    nodes_num=50,
    data_type="uniform",
    solver="lkh",
    train_samples_num=16,
    val_samples_num=16,
    test_samples_num=16,
    save_path="path/to/save/"
)

tsp_data_lkh.generate()
```