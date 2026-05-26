# FEM Integration in ML4CO-Kit

Dear FEM Team,

We would like to sincerely thank you for developing and open-sourcing [`FEM`](https://github.com/Fanerst/FEM).

`FEM` (Free-Energy Machine) introduces a physics-inspired approach for combinatorial optimization based on free-energy minimization, as published in *Nature Computational Science*. It provides an effective solver for graph-based CO problems such as Maximum Cut (MCut). The open-source release has significantly improved the accessibility and reproducibility of free-energy-based CO methods within modern research ecosystems.

We have recently integrated `FEM` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `FEM` is currently integrated into the **solver** component under the **graph** category. The integration converts task instances into adjacency representations, invokes the FEM algorithm with simulated annealing, and parses the returned solutions. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `FEM` within `ML4CO-Kit` is as follows:

|  Solver   |  Task  | Status |                 Code                 |
|:---------:|:------:|:------:|:------------------------------------:|
| FEMSolver |  MCut  | Supported | [ml4co_kit/solver/graph/lib/fem/mcut_fem.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/graph/lib/fem/mcut_fem.py) |
| FEMSolver |  MIS  | TODO |  |
| FEMSolver |  MCl  | TODO |  |
| FEMSolver |  MVC  | TODO |  |

Below are simple examples of how `FEMSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using FEM solver to generate MCut dataset.
"""
import numpy as np
from ml4co_kit import MCutWrapper, MCutGenerator, FEMSolver, GRAPH_TYPE

# Set the wrapper for MCut
mcut_wrapper = MCutWrapper()

# Set the generator for MCut
mcut_generator = MCutGenerator(
    graph_type=GRAPH_TYPE.BA,
    precision=np.float32,
    nodes_num=50,
)

# Set the solver for MCut
mcut_solver = FEMSolver(
    num_trials=100,
    num_steps=1000,
    device="cpu",
)

# Generate MCut data
mcut_wrapper.generate_w_to_txt(
    file_path="mcut_ba-50.txt",
    generator=mcut_generator,
    solver=mcut_solver,
    num_samples=100,
    num_threads=4,
    show_time=True,
)
```

```python
r"""
Example of using FEM solver to solve MCut dataset.
"""
from ml4co_kit import MCutWrapper, FEMSolver

# Set the wrapper for MCut
mcut_wrapper = MCutWrapper()

# Set the solver for MCut
mcut_solver = FEMSolver(num_trials=100, num_steps=1000, device="cpu")

# Read the dataset from pkl file
mcut_wrapper.from_pkl(
    file_path="test_dataset/graph/mcut/wrapper/mcut_ba-small_no-weighted_4ins.pkl",
    ref=True,
)

# Solve the dataset using FEM solver
mcut_wrapper.solve(solver=mcut_solver, num_threads=4, show_time=True)

# Evaluate the solution
print(mcut_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/fem](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/fem).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `FEM` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
