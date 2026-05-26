# RLD4CO (RLSA) Integration in ML4CO-Kit

Dear RLD4CO Team,

We would like to sincerely thank you for developing and open-sourcing [`RLD4CO`](https://github.com/Shengyu-Feng/RLD4CO).

`RLD4CO` introduces Regularized Langevin Dynamics for Combinatorial Optimization, providing an effective approach for solving graph-based CO problems such as Maximum Clique (MCl), Maximum Cut (MCut), and Maximum Independent Set (MIS). The open-source release has improved the accessibility and reproducibility of Langevin-dynamics-based CO solvers within modern research ecosystems.

We have recently integrated `RLD4CO` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `RLD4CO` is currently integrated into the **solver** component under the **graph** category. The integration converts task instances into adjacency representations, invokes the RLSA algorithm, and parses the returned solutions. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `RLD4CO` within `ML4CO-Kit` is as follows:

|   Solver   |  Task  | Status |                 Code                 |
|:----------:|:------:|:------:|:------------------------------------:|
| RLSASolver |  MCl   | Supported | [ml4co_kit/solver/graph/lib/rlsa/mcl_rlsa.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/graph/lib/rlsa/mcl_rlsa.py) |
| RLSASolver |  MCut  | Supported | [ml4co_kit/solver/graph/lib/rlsa/mcut_rlsa.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/graph/lib/rlsa/mcut_rlsa.py) |
| RLSASolver |  MIS   | Supported | [ml4co_kit/solver/graph/lib/rlsa/mis_rlsa.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/graph/lib/rlsa/mis_rlsa.py) |
| RLSASolver |  MVC  | TODO |  |

Below are simple examples of how `RLSASolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using RLSA solver to generate MIS dataset.
"""
import numpy as np
from ml4co_kit import MISWrapper, MISGenerator, RLSASolver, GRAPH_TYPE

# Set the wrapper for MIS
mis_wrapper = MISWrapper()

# Set the generator for MIS
mis_generator = MISGenerator(
    graph_type=GRAPH_TYPE.ER,
    precision=np.float32,
    nodes_num=700,
    nodes_num_upper=800,
)

# Set the solver for MIS
mis_solver = RLSASolver(
    rlsa_tau=0.01,
    rlsa_d=5,
    rlsa_k=200,
    rlsa_t=300,
    rlsa_device="cpu",
)

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
Example of using RLSA solver to solve MCut dataset.
"""
from ml4co_kit import MCutWrapper, RLSASolver

# Set the wrapper for MCut
mcut_wrapper = MCutWrapper()

# Set the solver for MCut
mcut_solver = RLSASolver(rlsa_device="cpu")

# Read the dataset from pkl file
mcut_wrapper.from_pkl(
    file_path="test_dataset/graph/mcut/wrapper/mcut_ba-small_no-weighted_4ins.pkl",
    ref=True,
)

# Solve the dataset using RLSA solver
mcut_wrapper.solve(solver=mcut_solver, num_threads=4, show_time=True)

# Evaluate the solution
print(mcut_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/rlsa](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/rlsa).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `RLD4CO` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
