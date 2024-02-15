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
|  TSP  |tsplib| ✔ | uniform | ✔ |
|  MIS  |satlib| ✔ | ER | 📆 |

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

### How to Use Evaluator (TSPLIBEvaluator as example)

```python
>>> from data4co.evaluate import TSPLIBEvaluator
>>> from data4co.solver import TSPLKHSolver, TSPConcordeSolver

# test LKH
>>> lkh_solver = TSPLKHSolver(lkh_scale=1e2)
>>> eva = TSPLIBEvaluator()
>>> eva.evaluate(lkh_solver)
           solved_costs       gt_costs          gaps
a280        2586.769648    2586.769648  0.000000e+00
att48      33523.708507   33523.708507  0.000000e+00
berlin52    7544.365902    7544.365902  3.616585e-14
ch130       6110.722200    6110.860950 -2.270541e-03
ch150       6530.902722    6532.280933 -2.109847e-02
eil101       640.211591     642.309536 -3.266252e-01
eil51        428.871756     429.983312 -2.585113e-01
eil76        544.369053     545.387552 -1.867479e-01
kroA100    21285.443182   21285.443182  0.000000e+00
kroC100    20750.762504   20750.762504  0.000000e+00
kroD100    21294.290821   21294.290821  3.416858e-14
lin105     14382.995933   14382.995933  0.000000e+00
pr1002    260047.681630  259066.663053  3.786742e-01
pr2392    383849.940441  378062.826191  1.530728e+00
pr76      108159.438274  108159.438274 -1.345413e-14
rd100       7910.396210    7910.396210  0.000000e+00
st70         677.109609     678.597452 -2.192526e-01
tsp225      3859.000000    3859.000000  0.000000e+00
AVG        50007.054444   49631.448887  4.971646e-02

# test concorde
>>> con_solver = TSPConcordeSolver(concorde_scale=1e2)
>>> eva.evaluate(con_solver)
           solved_costs       gt_costs          gaps
a280        2586.769648    2586.769648 -1.757974e-14
att48      33523.708507   33523.708507  2.170392e-14
berlin52    7544.365902    7544.365902  0.000000e+00
ch130       6110.722200    6110.860950 -2.270541e-03
ch150       6530.902722    6532.280933 -2.109847e-02
eil101       640.211591     642.309536 -3.266252e-01
eil51        428.871756     429.983312 -2.585113e-01
eil76        544.369053     545.387552 -1.867479e-01
kroA100    21285.443182   21285.443182 -1.709139e-14
kroC100    20750.762504   20750.762504  0.000000e+00
kroD100    21294.290821   21294.290821  3.416858e-14
lin105     14382.995933   14382.995933 -1.264680e-14
pr1002    259066.663053  259066.663053  0.000000e+00
pr2392    378062.696085  378062.826191 -3.441403e-05
pr76      108159.438274  108159.438274 -1.345413e-14
rd100       7910.396210    7910.396210 -3.449238e-14
st70         677.109609     678.597452 -2.192526e-01
tsp225      3859.000000    3859.000000  0.000000e+00
AVG        49631.039836   49631.448887 -5.636336e-02
```