## Data4CO

[![PyPi version](https://badgen.net/pypi/v/data4co/)](https://pypi.org/pypi/data4co/)
[![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdata4co%2Fjson)](https://pypi.python.org/pypi/data4co/)
[![Downloads](https://static.pepy.tech/badge/data4co)](https://pepy.tech/project/data4co)
[![GitHub stars](https://img.shields.io/github/stars/heatingma/Data4CO.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/heatingma/Data4CO/stargazers/) 

A data generator tool for Combinatorial Optimization (CO) problems, enabling customizable, diverse, and scalable datasets for benchmarking optimization algorithms.

### Current support

**data**
| Problem |     First      | Impl. | Second  | Impl. |      Third      | Impl. |
|:-------:|:--------------:|:-----:|:-------:|:-----:|:---------------:|:-----:|
|   TSP   | TSPLIB(Origin) |   ✔   | Uniform |   ✔   | Multi. Distros. |  📆   |
|   MIS   |     SATLIB     |   ✔   |  KaMIS  |  📆   |       --        |  --   |

**draw**
| Problem |     problem      | Impl. |     solution     | Impl. |
|:-------:|:----------------:|:-----:|:----------------:|:-----:|
|   TSP   | draw_tsp_problem |   ✔   | draw_tsp_soluton |   ✔   |
|   MIS   | draw_mis_problem |  📆   | draw_mis_soluton |  📆   |

**evaluator**
| Problem |     Base     | Impl. |      First      | Impl. | Second  | Impl. |      Third      | Impl. |
|:-------:|:------------:|:-----:|:---------------:|:-----:|:-------:|:-----:|:---------------:|:-----:|
|   TSP   | TSPEvaluator |   ✔   | TSPLIB(Origin)  |   ✔   | Uniform |   ✔   | Multi. Distros. |  📆   |
|   MIS   | MISEvaluator |  📆   | SATLIBEvaluator |  📆   |   --    |  --   |       --        |  --   |

**generator**
| Problem |  Type1  | Impl. |  Type2   | Impl. |  Type3  | Impl. |  Type4   | Impl. |
|:-------:|:-------:|:-----:|:--------:|:-----:|:-------:|:-----:|:--------:|:-----:|
|   TSP   | uniform |   ✔   | gaussian |   ✔   | cluster |   ✔   | w/regret |   ✔   |
|   MIS   |   ER    |   ✔   |    BA    |   ✔   |   HK    |   ✔   |    WS    |   ✔   |

**solver**
| Problem |   Base    | Impl. | First | Impl. |  Second  | Impl. |      Third      | Impl. |
|:-------:|:---------:|:-----:|:-----:|:-----:|:--------:|:-----:|:---------------:|:-----:|
|   TSP   | TSPSolver |   ✔   |  LKH  |   ✔   | Concorde |   ✔   | Concorde(Large) |   ✔   |
|   MIS   | MISSolver |   ✔   | KaMIS |   ✔   |  Gurobi  |   ✔   |       --        |  --   |

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
requests>=2.31.0
aiohttp>=3.9.3
async_timeout>=4.0.3
```
If you want to obtain complete data4co's functions, like drawing, the following packages need to be installed by ``pip``:
```
matplotlib>=3.7.5
```

**PyPI**
It is very convenient to directly use the following commands
```
pip install data4co
```

### Solver Example

```python
from data4co.solver import TSPLKHSolver

tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=500)
tsp_lkh_solver.from_txt("path/to/read/file.txt")
tsp_lkh_solver.solve()
tsp_lkh_solver.evaluate()
tsp_lkh_solver.to_txt("path/to/write/file.txt")
```

### Generator Example

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

### Evaluator Example

```python
>>> from data4co.evaluate import TSPLIBOriginEvaluator
>>> from data4co.solver import TSPLKHSolver, TSPConcordeSolver

# test LKH
>>> lkh_solver = TSPLKHSolver(scale=1)
>>> eva = TSPLIBOriginEvaluator()
>>> eva.evaluate(lkh_solver, norm="EUC_2D")
           solved_costs       gt_costs          gaps
att48      33523.708507   33523.708507  0.000000e+00
eil51        429.983312     429.983312  0.000000e+00
berlin52    7544.365902    7544.365902  3.616585e-14
st70         678.557469     678.597452 -5.892021e-03
pr76      108159.438274  108159.438274 -1.345413e-14
kroC100    20750.762504   20750.762504  0.000000e+00
eil101       642.856874     642.309536  8.521414e-02
lin105     14382.995933   14382.995933  0.000000e+00
ch130       6110.900592    6110.860950  6.487238e-04
ch150       6530.902722    6532.280933 -2.109847e-02
tsp225      3859.000000    3859.000000  0.000000e+00
a280        2588.301213    2586.769648  5.920765e-02
pr1002    260277.189980  259066.663053  4.672646e-01
pr2392    384469.093320  378062.826191  1.694498e+00
AVG        60710.575472   60166.468728  1.628459e-01

>>> eva.evaluate(lkh_solver, norm="GEO")
           solved_costs  gt_costs      gaps
ulysses16        6859.0    6859.0  0.000000
ulysses22        7013.0    7013.0  0.000000
gr96            55209.0   55209.0  0.000000
gr202           40160.0   40160.0  0.000000
gr666          295012.0  294358.0  0.222178
AVG             80850.6   80719.8  0.044436

# test concorde
>>> con_solver = TSPConcordeSolver(scale=1)
>>> eva.evaluate(con_solver, norm="EUC_2D")
           solved_costs       gt_costs          gaps
att48      33523.708507   33523.708507  2.170392e-14
eil51        429.117939     429.983312 -2.012573e-01
berlin52    7544.365902    7544.365902  0.000000e+00
st70         678.583751     678.597452 -2.019036e-03
pr76      108159.438274  108159.438274 -1.345413e-14
kroC100    20750.762504   20750.762504  0.000000e+00
eil101       641.690973     642.309536 -9.630290e-02
lin105     14382.995933   14382.995933  0.000000e+00
ch130       6110.739012    6110.860950 -1.995428e-03
ch150       6532.280933    6532.280933  0.000000e+00
tsp225      3859.000000    3859.000000  0.000000e+00
a280        2587.930486    2586.769648  4.487600e-02
pr1002    259066.663053  259066.663053 -1.123411e-14
pr2392    378062.826191  378062.826191  0.000000e+00
AVG        60166.435961   60166.468728 -1.833562e-02

>>> eva.evaluate(con_solver, norm="GEO")
           solved_costs  gt_costs  gaps
ulysses16        6859.0    6859.0   0.0
ulysses22        7013.0    7013.0   0.0
gr96            55209.0   55209.0   0.0
gr202           40160.0   40160.0   0.0
gr666          294358.0  294358.0   0.0
AVG             80719.8   80719.8   0.0
```

### Draw Example

```python
>>> from data4co.solver import TSPConcordeSolver
>>> from data4co.draw.tsp import draw_tsp_solution, draw_tsp_problem

# use TSPConcordeSolver to solve the problem
>>> solver = TSPConcordeSolver(scale=1)
>>> solver.from_tsp("docs/kroA150.tsp")
>>> solver.solve(norm="EUC_2D")

# draw
draw_tsp_problem(
    save_path="docs/kroA150_problem.png",
    points=solver.ori_points,
)
draw_tsp_solution(
    save_path="docs/kroA150_solution.png",
    points=solver.ori_points,
    tours=solver.tours
)
```
<details>
<summary>Click to view the drawing results</summary>
<img src="docs/kroA150_problem.png" width="35%" alt="" />
<img src="docs/kroA150_solution.png" width="35%" alt="" />
</details>
