<img src="docs/assets/ml4co-kit-logo.png" alt="ML4CO-Kit" width="800"/>

[![PyPi version](https://badgen.net/pypi/v/ml4co-kit/)](https://pypi.org/pypi/ml4co_kit/) [![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fml4co_kit%2Fjson)](https://pypi.python.org/pypi/ml4co-kit/) [![Downloads](https://static.pepy.tech/badge/ml4co-kit)](https://pepy.tech/project/ml4co-kit) [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/ML4CO-Kit.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/ML4CO-Kit/stargazers/)

`ML4CO-Kit` is an in-development toolkit for machine learning practices on combinatorial optimization problems, which is a by-product of our research for a unified modular framework that integrates existing ML4CO practices, minimizing disparities among methods and supporting the investigations via in-depth analysis and transparent ablation. 

This repository focuses on the supporting code for method development instead of implementing core technologies, which will be presented in the future in our full implementation and organization. `ML4CO-Kit` has the following features:

* The skeleton of framework organization for ML4CO projects;
* Implemented base classes that facilitate method development;
* Mainstream traditional solver baselines and reference solution acquisition;
* Data generation of various distributions;
* Problem and solution visualization;
* Evaluators for different problems.
* Common post-processing algorithms

| Problem |                Data                |    solver     |      Supervision      | evaluator |      visualization      |
| :-----: | :--------------------------------: | :-----------: | :-------------------: | :-------: | :---------------------: |
|  ATSP   |    SAT, HCP       | LKH |       Solution        |     ✔     | -- |
|  CVRP   |    Uniform, Gaussian, VRPLIB       | PyVRP, LKH, HGS |       Solution        |     ✔     | Problem Graph, Solution |
|   MC   |       SATLIB, ER, BA, HK, WS       | Gurobi |       Solution        |     ✔     | Problem Graph, Solution |
|   MCl   |       SATLIB, ER, BA, HK, WS       | Gurobi |       Solution        |     ✔     | Problem Graph, Solution |
|   MIS   |       SATLIB, ER, BA, HK, WS       | KaMIS, Gurobi |       Solution        |     ✔     | Problem Graph, Solution |
|   MVC   |       SATLIB, ER, BA, HK, WS       | Gurobi |       Solution        |     ✔     | Problem Graph, Solution |
|   TSP   | Uniform, Gaussian, Cluster, TSPLIB | LKH, Concorde, Concorde-Large, GA-EAX, GA-EAX-Large | Solution, Edge Regret |     ✔     | Problem Graph, Solution |

We are still enriching the library and we welcome any contributions/ideas/suggestions from the community. A comprehensive modular framework built upon this library that integrates core ML4CO technologies is coming soon.

## Installation

You can install the stable release on PyPI:

```bash
$ pip install ml4co_kit
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

## Usage Examples

### Solve with Traditional Solver Baselines

We provide base classes with a user-friendly approach for implementing traditional and learning-based solvers. Taking `TSPSolver` as an example, it includes functionalities for data input and output, as well as an evaluation function. The solver supports different data inputs, such as Numpy arrays and .txt and .tsp files. The outputs can be saved to corresponding types of files as needed. Additionally, the solver offers an evaluation function, by which users can quickly obtain the average tour length, average gap, and standard deviation of the test dataset. Traditional solvers are directly incorporated in our library inheriting `TSPSolver`.

```python
>>> from ml4co_kit.solver import TSPLKHSolver

# initialization
>>> tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=500)

# input instances and reference solutions by a .txt file
>>> tsp_lkh_solver.from_txt("path/to/read/tsp500_concorde.txt")

# lkh solving
>>> tsp_lkh_solver.solve()

# evaluate
>>> tsp_lkh_solver.evaluate(calculate_gap=True)
(16.583557978549532, 0.21424058722308548, 0.09031979488795724)

# save solving results
>>> tsp_lkh_solver.to_txt("path/to/write/tsp500_lkh.txt")
```

### Data Generation

```python
from ml4co_kit import TSPDataGenerator

# initialization
tsp_data_lkh = TSPDataGenerator(
    num_threads=8,
    nodes_num=50,
    data_type="uniform",
    solver="LKH",
    train_samples_num=16,
    val_samples_num=16,
    test_samples_num=16,
    save_path="path/to/save/"
)

# generate
tsp_data_lkh.generate()
```

### Evaluate

```python
>>> from ml4co_kit.evaluate import TSPLIBOriEvaluator
>>> from ml4co_kit.solver import TSPLKHSolver

>>> lkh_solver = TSPLKHSolver(scale=1)
>>> evaluator = TSPLIBOriEvaluator()
>>> evaluator.evaluate(lkh_solver, norm="EUC_2D")
           solved_costs      ref_costs          gaps
eil51        429.983312     429.983312  0.000000e+00
berlin52    7544.365902    7544.365902  3.616585e-14
st70         677.881928     678.597452 -1.054416e-01
eil76        544.837795     545.387552 -1.008012e-01
pr76      108159.438274  108159.438274 -1.345413e-14
kroA100    21285.443182   21285.443182  0.000000e+00
kroC100    20750.762504   20750.762504  0.000000e+00
kroD100    21294.290821   21294.290821  3.416858e-14
rd100       7910.396210    7910.396210  0.000000e+00
eil101       642.244814     642.309536 -1.007642e-02
lin105     14382.995933   14382.995933  0.000000e+00
ch130       6110.739012    6110.860950 -1.995428e-03
ch150       6532.280933    6532.280933 -2.784616e-14
tsp225      3859.000000    3859.000000  0.000000e+00
a280        2587.930486    2586.769648  4.487600e-02
pr1002    259066.663053  259066.663053  0.000000e+00
pr2392    378062.826191  378062.826191  0.000000e+00
AVG        50578.945903   50578.963027 -1.020227e-02

>>> evaluator.evaluate(lkh_solver, norm="GEO")
           solved_costs    ref_costs          gaps
ulysses16     74.108736    74.108736  1.917568e-14
ulysses22     75.665149    75.665149  3.756248e-14
gr96         512.309380   512.309380  0.000000e+00
gr202        549.998070   549.998070 -8.268163e-14
gr666       3843.137961  3952.535702 -2.767786e+00
AVG         1011.043859  1032.923407 -5.535573e-01
```

### Visualization

#### CVRP

<p>
<img src="docs/assets/cvrp_problem.png" width="35%" alt="" />
<img src="docs/assets/cvrp_solution.png" width="35%" alt="" />
</p>

#### MCl

<p>
<img src="docs/assets/mcl_problem.png" width="35%" alt="" />
<img src="docs/assets/mcl_solution.png" width="35%" alt="" />
</p>

#### MCut

<p>
<img src="docs/assets/mcut_problem.png" width="35%" alt="" />
<img src="docs/assets/mcut_solution.png" width="35%" alt="" />
</p>

#### MIS

<p>
<img src="docs/assets/mis_problem.png" width="35%" alt="" />
<img src="docs/assets/mis_solution.png" width="35%" alt="" />
</p>

#### MVC

<p>
<img src="docs/assets/mvc_problem.png" width="35%" alt="" />
<img src="docs/assets/mvc_solution.png" width="35%" alt="" />
</p>

#### TSP

<p>
<img src="docs/assets/tsp_problem.png" width="35%" alt="" />
<img src="docs/assets/tsp_solution.png" width="35%" alt="" />
</p>


### Develop ML4CO Algorithms

Please refer to `ml4co_kit/learning` for the base classes that facilitate a quick establishment of a ML4CO project. You can easily build a project by inheriting the base classes and additionally implement task-specific and methodology-specific functions according to [ML4CO Organization](#ML4CO Organization:). We provide an minimalistic exmple of build a simple ML4TSP project in `docs/project_example`.
