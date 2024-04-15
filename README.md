<img src="docs/assets/ml4co-kit-logo.png" alt="ML4CO-Kit" width="800"/>

[![PyPi version](https://badgen.net/pypi/v/data4co/)](https://pypi.org/pypi/data4co/) [![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdata4co%2Fjson)](https://pypi.python.org/pypi/data4co/) [![Downloads](https://static.pepy.tech/badge/data4co)](https://pepy.tech/project/data4co) [![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/ML4CO-Kit.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/ML4CO-Kit/stargazers/) 

`ML4CO-Kit` is a in-development toolkit for machine learning practices on combinatorial optimization problems, which is a by-product of our research for a unified modular framework that integrates existing ML4CO practices, minimizing disparities among methods and supporting the investigations via in-depth analysis and transparent ablation. 

This reposity focuses on the supporting code for method development instead of implementing core technologies, which will be presented in the future  in our full implementation and organization. `ML4CO-Kit` has the following features:

* The skeleton of framework organization for ML4CO projects;
* Implemented base classes that facilitate method development;
* Mainstream traditional solver baselines and reference solution acquisition;
* Data generation of various distributions;
* Problem and solution visualization;
* Evaluators for different problems.

| Problem |                Data                |    solver     |      Supervision      | evaluator |      visualization      |
| :-----: | :--------------------------------: | :-----------: | :-------------------: | :-------: | :---------------------: |
|   TSP   | Uniform, Gaussian, Cluster, TSPLIB | LKH, Concorde | Solution, Edge Regret |     ✔     | Problem Graph, Solution |
|   MIS   |       SATLIB, ER, BA, HK, WS       | KaMIS, Gurobi |       Solution        |     ✔     | Problem Graph, Solution |

###### ML4CO Organization:

<img src="docs/assets/organization.jpg" alt="Organization" width="800"/>

We are still enriching the library and we welcome any contributions/ideas/suggestions from the community. A comprehensive modular framework built upon this library that integrates core ML4CO technologies coming soon.

## Installation

You can install the stable release on PyPI:

```bash
$ pip install ml4co_kit
```

or get the latest version by running:

```bash
$ pip install -U https://github.com/Thinklab-SJTU/ML4CO-Kit/archive/master.zip # with --user for user install (no root)
```

The following packages are required, and shall be automatically installed by ``pip``:

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

To ensure you have access to all functions, such as visualization, you'll need to install the following packages using `pip`:

```bash
$ pip install matplotlib>=3.7.5
```

## Usage Examples

### Solve with  Traditional Solver Baselines

We provide base classes that offer a user-friendly approach for implementing traditional and learning-based solvers. Taking `TSPSolver` as an example, which includes functionalities for data input and output, as well as an evaluation function. The solver supports different data inputs, such as Numpy arrays and .txt and .tsp files. The outputs can be saved to corresponding types of files as needed. Additionally, the solver offers an evaluation function, by which users can quickly obtain the average tour length, average gap, and standard deviation of the test dataset. Traditional solvers are directly incorporated in our library inheriting `TSPSolver`.

```python
>>> from ml4co_kit.solver import TSPLKHSolver

# initialization
>>> tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=500)

# input instances and reference solutions by a .txt file
>>> tsp_lkh_solver.from_txt("path/to/read/tsp500_concorde.txt")

# lkh solving
>>> tsp_lkh_solver.solve()

# evaluate
>>> tsp_lkh_solver.evaluate(caculate_gap=True)
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
    solver="lkh",
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
           solved_costs       ref_costs          gaps
att48      33523.708507   33523.708507  0.000000e+00
eil51        429.983312     429.983312  0.000000e+00
berlin52    7544.365902    7544.365902  3.616585e-14
st70         678.557469     678.597452 -5.892021e-03
eil76        545.229738     545.387552 -2.893612e-02
pr76      108159.438274  108159.438274 -1.345413e-14
kroA100    21285.443182   21285.443182  0.000000e+00
kroC100    20750.762504   20750.762504  0.000000e+00
kroD100    21294.290821   21294.290821  3.416858e-14
rd100       7910.396210    7910.396210  0.000000e+00
eil101       642.856874     642.309536  8.521414e-02
lin105     14382.995933   14382.995933  0.000000e+00
ch130       6110.900592    6110.860950  6.487238e-04
ch150       6530.902722    6532.280933 -2.109847e-02
tsp225      3859.000000    3859.000000  0.000000e+00
a280        2588.301213    2586.769648  5.920765e-02
pr1002    260277.189980  259066.663053  4.672646e-01
pr2392    384469.093320  378062.826191  1.694498e+00
AVG        50054.634253   49631.448887  1.250504e-01

>>> eva.evaluate(lkh_solver, norm="GEO")
           solved_costs  ref_costs      gaps
ulysses16        6859.0    6859.0  0.000000
ulysses22        7013.0    7013.0  0.000000
gr96            55209.0   55209.0  0.000000
gr202           40160.0   40160.0  0.000000
gr666          295012.0  294358.0  0.222178
AVG             80850.6   80719.8  0.044436
```

### Visualization

#### TSP

```python
from data4co.solver import TSPConcordeSolver
from data4co.draw.tsp import draw_tsp_solution, draw_tsp_problem

# use TSPConcordeSolver to solve the problem
solver = TSPConcordeSolver(scale=1)
solver.from_tsp("examples/tsp/kroA150.tsp")
solver.solve(norm="EUC_2D")

# draw
draw_tsp_problem(
    save_path="docs/assets/kroA150_problem.png",
    points=solver.ori_points,
)
draw_tsp_solution(
    save_path="docs/assets/kroA150_solution.png",
    points=solver.ori_points,
    tours=solver.tours
)
```

Visualization Results:

<p>
<img src="docs/assets/kroA150_problem.png" width="35%" alt="" />
<img src="docs/assets/kroA150_solution.png" width="35%" alt="" />
</p>

#### MIS

```python
from data4co.solver import KaMISSolver
from data4co import draw_mis_problem, draw_mis_solution

# use KaMISSolver to solve the problem
mis_solver = KaMISSolver()
mis_solver.solve(src="examples/mis_example")

# draw
draw_mis_problem(
    save_path="docs/assets/mis_problem.png", 
    gpickle_path="examples/mis/mis_example.gpickle"
)
draw_mis_solution(
    save_path="docs/mis_solution.png",
    gpickle_path="examples/mis/mis_example.gpickle",
    result_path="examples/mis/solve/mis_example_unweighted.result"
)
```

Visualization Results:

<p>
<img src="docs/assets/mis_problem.png" width="35%" alt="" />
<img src="docs/assets/mis_solution.png" width="35%" alt="" />
</p>

### Develop ML4CO Algorithms

Please refer to `ml4co_kit/learning` for the base classes that facilitate a quick establishment of a ML4CO project. You can easily build a project by inheriting the base classes and additionally implement task-specific and methodology-specific functions according to [ML4CO Organization](#ML4CO Organization:). We provide an minimalistic exmple of build a simple ML4TSP project in `docs/project_example`.

