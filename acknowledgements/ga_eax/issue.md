# GA-EAX Integration in ML4CO-Kit

Dear GA-EAX Team,

We would like to sincerely thank you for developing and open-sourcing [`GA-EAX`](https://github.com/nagata-yuichi/GA-EAX).

`GA-EAX` is a powerful genetic algorithm using Edge Assembly Crossover (EAX) for the Traveling Salesman Problem (TSP). It is widely recognized as one of the most effective heuristic solvers for TSP and has been adopted extensively in CO research and benchmarking. Making this implementation publicly available has significantly improved reproducibility and accessibility for modern ML4CO pipelines, benchmarking systems, and dataset generation workflows.

We have recently integrated `GA-EAX` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `GA-EAX` is currently integrated into the **solver** component under the **routing** category. The integration converts task instances to TSPLIB format, invokes the bundled GA-EAX executable, and parses the returned tours. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `GA-EAX` within `ML4CO-Kit` is as follows:

|   Solver    |  Task  | Status |                 Code                 |
|:-----------:|:------:|:------:|:------------------------------------:|
| GAEAXSolver |  TSP   | Supported | [ml4co_kit/solver/routing/lib/ga_eax/tsp_ga_eax.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/ga_eax/tsp_ga_eax.py) |

Below are simple examples of how `GAEAXSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using GA-EAX solver to generate TSP dataset.
"""
import numpy as np
from ml4co_kit import TSPWrapper, TSPGenerator, GAEAXSolver, TSP_TYPE

# Set the wrapper for TSP
tsp_wrapper = TSPWrapper()

# Set the generator for TSP
tsp_generator = TSPGenerator(
    distribution_type=TSP_TYPE.UNIFORM,
    precision=np.float32,
    nodes_num=50,
)

# Set the solver for TSP
tsp_solver = GAEAXSolver(
    ga_eax_population_num=100,
    ga_eax_offspring_num=30,
    use_large_solver=False,
)

# Generate TSP data
tsp_wrapper.generate_w_to_txt(
    file_path="tsp50_uniform.txt",
    generator=tsp_generator,
    solver=tsp_solver,
    num_samples=100,
    num_threads=4,
    show_time=True,
)
```

```python
r"""
Example of using GA-EAX solver to solve TSP dataset.
"""
from ml4co_kit import TSPWrapper, GAEAXSolver

# Set the wrapper for TSP
tsp_wrapper = TSPWrapper()

# Set the solver for TSP
tsp_solver = GAEAXSolver(
    ga_eax_scale=1e5,
    ga_eax_max_trials=1,
    ga_eax_population_num=100,
    ga_eax_offspring_num=30,
    ga_eax_show_info=False,
    use_large_solver=False,
)

# Read the dataset from txt file
tsp_wrapper.from_txt(file_path="tsp50_uniform_16ins.txt", ref=True)

# Solve the dataset using GA-EAX solver
tsp_wrapper.solve(solver=tsp_solver, num_threads=4, show_time=True)

# Evaluate the solution
print(tsp_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository, license, and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/ga_eax](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/ga_eax).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `GA-EAX` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
