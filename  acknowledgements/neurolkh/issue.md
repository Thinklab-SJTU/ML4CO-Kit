# NeuroLKH Integration in ML4CO-Kit

Dear NeuroLKH Team,

We would like to sincerely thank you for developing and open-sourcing [`NeuroLKH`](https://github.com/liangxinedu/NeuroLKH).

`NeuroLKH` combines deep learning with the Lin-Kernighan-Helsgaun (LKH) heuristic to solve the Traveling Salesman Problem (TSP), achieving strong performance by guiding LKH search with learned candidate sets. The open-source release has significantly improved the accessibility and reproducibility of learning-augmented TSP solvers within modern research ecosystems.

We have recently integrated `NeuroLKH` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `NeuroLKH` is currently integrated into the **solver** component under the **routing** category. The integration converts TSP task instances into TSPLIB format, invokes the NeuroLKH neural-guided LKH pipeline in batch mode, and parses the returned tours. The integration builds upon the bundled LKH executable and pretrained NeuroLKH models. It is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `NeuroLKH` within `ML4CO-Kit` is as follows:

|     Solver     |  Task  | Status |                 Code                 |
|:--------------:|:------:|:------:|:------------------------------------:|
| NeuroLKHSolver |  TSP   | Supported | [ml4co_kit/solver/routing/lib/neurolkh/tsp_neurolkh.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/neurolkh/tsp_neurolkh.py) |
| NeuroLKHSolver |  CVRP   | TODO |
| NeuroLKHSolver |  CVRPTW   | TODO |


> **Note on batch solving:** `NeuroLKHSolver` leverages neural guidance through `batch_solve` for TSP instances. When generating or solving datasets, specify `batch_size` in the wrapper API to enable efficient batched NeuroLKH inference.

Below are simple examples of how `NeuroLKHSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using NeuroLKH solver to generate TSP dataset (batch mode).
"""
import numpy as np
from ml4co_kit import TSPWrapper, TSPGenerator, NeuroLKHSolver, TSP_TYPE

# Set the wrapper for TSP
tsp_wrapper = TSPWrapper()

# Set the generator for TSP
tsp_generator = TSPGenerator(
    distribution_type=TSP_TYPE.UNIFORM,
    precision=np.float32,
    nodes_num=500,
)

# Set the solver for TSP
tsp_solver = NeuroLKHSolver(
    lkh_scale=int(1e6),
    lkh_max_trials=500,
    lkh_runs=1,
    lkh_seed=1234,
    neurolkh_device="cuda",
    neurolkh_tree_cands_num=10,
    neurolkh_search_cands_num=5,
    neurolkh_initial_period=15,
    neurolkh_sparse_factor=20,
)

# Generate TSP data using batch solving
tsp_wrapper.generate_w_to_txt(
    file_path="tsp50_uniform.txt",
    generator=tsp_generator,
    solver=tsp_solver,
    num_samples=100,
    num_threads=1,
    batch_size=4,
    show_time=True,
)
```

```python
r"""
Example of using NeuroLKH solver to solve TSP dataset (batch mode).
"""
from ml4co_kit import TSPWrapper, NeuroLKHSolver

# Set the wrapper for TSP
tsp_wrapper = TSPWrapper()

# Set the solver for TSP
tsp_solver = NeuroLKHSolver(
    lkh_scale=int(1e6),
    lkh_max_trials=500,
    lkh_runs=1,
    lkh_seed=1234,
    neurolkh_device="cuda",
    neurolkh_tree_cands_num=10,
    neurolkh_search_cands_num=5,
    neurolkh_initial_period=15,
    neurolkh_sparse_factor=20,
)

# Read the dataset from txt file
tsp_wrapper.from_txt(file_path="tsp500_uniform_4ins.txt", ref=True)

# Solve the dataset using NeuroLKH solver (batch mode)
tsp_wrapper.solve(solver=tsp_solver, batch_size=4, show_time=True)

# Evaluate the solution
print(tsp_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/neurolkh](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/neurolkh).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `NeuroLKH` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
