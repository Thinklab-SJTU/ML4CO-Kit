# pygmtools Integration in ML4CO-Kit

Dear pygmtools Team,

We would like to sincerely thank you for developing and open-sourcing [`pygmtools`](https://github.com/Thinklab-SJTU/pygmtools).

`pygmtools` provides a unified and comprehensive set of graph matching solvers accessible from Python, supporting various backends (numpy, pytorch, jittor, paddle, tensorflow). It has become a standard toolkit for graph matching, graph edit distance, and quadratic assignment problems in the CO and ML research communities.

We have recently integrated `pygmtools` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `pygmtools` is currently integrated into the **solver** component under the **QAP** category. The integration constructs affinity matrices from task instances, invokes pygmtools solvers, and parses the returned matching solutions. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

* **Task:** standardized abstractions and formulations for COPs.
* **Generator:** unified pipelines for synthetic instance generation.
* **Solver:** interfaces for integrating traditional, heuristic, and learning-based solvers.
* **Optimizer:** to further optimize the initial solution obtained by the solver.
* **Wrapper:** handling data R&W, task storage, parallelized generation and solving.

Beyond these five levels, we are also actively building a broader ecosystem for ML4CO research, including datasets (maintained at [huggingface.co/ML4CO](https://huggingface.co/ML4CO)), benchmarks (such as [ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)), and learning frameworks (such as [COExpander](https://github.com/Thinklab-SJTU/COExpander), [M2GenCO](https://github.com/Thinklab-SJTU/M2GenCO)). All of these projects are currently under active development and continuous maintenance.

ML4CO-Kit can be installed directly via:
```bash
pip install ml4co-kit
```

---

## Support Status and Example Usage

At the current stage, the support status of `pygmtools` within `ML4CO-Kit` is as follows:

|   Solver   |  Task  | Status |                 Code                 |
|:----------:|:------:|:------:|:------------------------------------:|
| PyGMSolver |  GM    | Supported | [ml4co_kit/solver/qap/lib/pygm/gm_pygm.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/qap/lib/pygm/gm_pygm.py) |
| PyGMSolver |  GED   | Supported | [ml4co_kit/solver/qap/lib/pygm/ged_pygm.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/qap/lib/pygm/ged_pygm.py) |
| PyGMSolver |  KQAP  | TODO |  |

Below are simple examples of how `PyGMSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using PyGMSolver to generate GM dataset.
"""
from ml4co_kit import GMWrapper, GMGenerator, PyGMSolver, GM_TYPE

# Set the wrapper for GM
gm_wrapper = GMWrapper()

# Set the generator for GM
gm_generator = GMGenerator(
    graph_matching_type=GM_TYPE.ISO,
    nodes_num=10,
    graph_num=100,
)

# Set the solver for GM
gm_solver = PyGMSolver()

# Generate GM data
gm_wrapper.generate_w_to_pkl(
    file_path="gm_iso_100ins.pkl",
    generator=gm_generator,
    solver=gm_solver,
    num_samples=100,
    show_time=True,
)
```

```python
r"""
Example of using PyGMSolver to solve GM dataset.
"""
from ml4co_kit import GMWrapper, PyGMSolver

# Set the wrapper for GM
gm_wrapper = GMWrapper()

# Set the solver for GM
gm_solver = PyGMSolver()

# Read the dataset from pkl file
gm_wrapper.from_pkl(
    file_path="test_dataset/qap/gm/wrapper/gm_iso_4ins.pkl",
    ref=True,
)

# Solve the dataset using PyGMSolver
gm_wrapper.solve(solver=gm_solver, show_time=True)

# Evaluate the solution
print(gm_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository, license, and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/pygmtools](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/pygmtools).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `pygmtools` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
