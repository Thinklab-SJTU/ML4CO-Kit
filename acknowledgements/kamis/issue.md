# KaMIS Integration in ML4CO-Kit

Dear KaMIS Team,

We would like to sincerely thank you for developing and open-sourcing [`KaMIS`](https://github.com/KarlsruheMIS/KaMIS).

`KaMIS` (Karlsruhe Maximum Independent Sets) is a suite of algorithms for computing large independent sets on huge sparse graphs. It includes SOTA kernelization techniques and local search algorithms, and is widely adopted in CO research and benchmarking for the Maximum Independent Set (MIS) problem. The open-source release has significantly improved the accessibility and reproducibility of high-performance MIS solvers within modern research ecosystems.

We have recently integrated `KaMIS` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `KaMIS` is currently integrated into the **solver** component under the **graph** category. The integration converts task instances to the METIS graph format, invokes the bundled KaMIS executable, and parses the returned independent sets. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `KaMIS` within `ML4CO-Kit` is as follows:

|   Solver    |  Task  | Status |                 Code                 |
|:-----------:|:------:|:------:|:------------------------------------:|
| KaMISSolver |  MIS   | Supported | [ml4co_kit/solver/graph/lib/kamis/mis_kamis.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/graph/lib/kamis/mis_kamis.py) |

Below are simple examples of how `KaMISSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using KaMIS solver to generate MIS dataset.
"""
import numpy as np
from ml4co_kit import MISWrapper, MISGenerator, KaMISSolver, GRAPH_TYPE

# Set the wrapper for MIS
mis_wrapper = MISWrapper()

# Set the generator for MIS
mis_generator = MISGenerator(
    distribution_type=GRAPH_TYPE.ER,
    precision=np.float32,
    nodes_num_scale=(700, 800),
    er_prob=0.15,
    node_weighted=False
)

# Set the solver for MIS
mis_solver = KaMISSolver(kamis_time_limit=10.0)

# Generate MIS data
mis_wrapper.generate_w_to_txt(
    file_path="mis_er-700-800.txt",
    generator=mis_generator,
    solver=mis_solver,
    num_samples=100,
    num_threads=4,
    show_time=True,
)
```

```python
r"""
Example of using KaMIS solver to solve MIS dataset.
"""
from ml4co_kit import MISWrapper, KaMISSolver

# Set the wrapper for MIS
mis_wrapper = MISWrapper()

# Set the solver for MIS
mis_solver = KaMISSolver(kamis_time_limit=10.0)

# Read the dataset from txt file
mis_wrapper.from_txt(file_path="mis_er-700-800_no-weighted_4ins.txt", ref=True)

# Solve the dataset using KaMIS solver
mis_wrapper.solve(solver=mis_solver, num_threads=4, show_time=True)

# Evaluate the solution
print(mis_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository, license, and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/kamis](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/kamis).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `KaMIS` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
