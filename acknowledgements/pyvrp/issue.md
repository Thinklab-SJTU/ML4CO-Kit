# PyVRP Integration in ML4CO-Kit

Dear PyVRP Team,

We would like to sincerely thank you for developing and open-sourcing [`PyVRP`](https://github.com/PyVRP/PyVRP).

`PyVRP` is a high-performance VRP solver package that builds on the Hybrid Genetic Search (HGS) framework. It supports a broad range of vehicle routing problem variants, including CVRP, CVRPTW, and problems with backhauls, distance limits, and their combinations. The clean Python API, pip-installable package, and active community maintenance have significantly improved the accessibility of SOTA VRP solving within modern Python-based research ecosystems.

We have recently integrated `PyVRP` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `PyVRP` is currently integrated into the **solver** component under the **routing** category. The integration converts task instances into the format expected by PyVRP, invokes the solver, and parses the returned routes. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `PyVRP` within `ML4CO-Kit` is as follows:

> **Note on Open VRP variants:** In `ML4CO-Kit`, the "open" dimension (i.e., vehicles do not need to return to the depot) is handled by a `cvrp_open` parameter within each task, rather than introducing additional task types. For example, `CVRPTask(cvrp_open=True)` represents the OVRP. This applies uniformly to all CVRP variants listed below.

|    Solver    |   Task   | Status |                 Code                 |
|:------------:|:--------:|:------:|:------------------------------------:|
| PyVRPSolver  |  CVRP    | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrp_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrp_pyvrp.py) |
| PyVRPSolver  |  CVRPB   | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrpb_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrpb_pyvrp.py) |
| PyVRPSolver  |  CVRPBL  | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrpbl_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrpbl_pyvrp.py) |
| PyVRPSolver  | CVRPBLTW | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrpbltw_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrpbltw_pyvrp.py) |
| PyVRPSolver  |  CVRPBTW | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrpbtw_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrpbtw_pyvrp.py) |
| PyVRPSolver  |  CVRPL   | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrpl_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrpl_pyvrp.py) |
| PyVRPSolver  | CVRPLTW  | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrpltw_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrpltw_pyvrp.py) |
| PyVRPSolver  |  CVRPTW  | Supported | [ml4co_kit/solver/routing/lib/pyvrp/cvrptw_pyvrp.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/pyvrp/cvrptw_pyvrp.py) |

Below are simple examples of how `PyVRPSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using PyVRP solver to generate CVRP dataset.
"""
import numpy as np
from ml4co_kit import CVRPWrapper, CVRPGenerator, PyVRPSolver, CVRP_TYPE

# Set the wrapper for CVRP
cvrp_wrapper = CVRPWrapper()

# Set the generator for CVRP
cvrp_generator = CVRPGenerator(
    cvrp_open=False,
    distribution_type=CVRP_TYPE.UNIFORM,
    precision=np.float32,
    nodes_num=50,
    min_demand=1,
    max_demand=9,
    min_capacity=40,
    max_capacity=40,
)

# Set the solver for CVRP
cvrp_solver = PyVRPSolver(time_limit=1.0, seed=1234)

# Generate CVRP data
cvrp_wrapper.generate_w_to_txt(
    file_path="cvrp50_uniform.txt",
    generator=cvrp_generator,
    solver=cvrp_solver,
    num_samples=100,
    num_threads=4,
    show_time=True,
)
```

```python
r"""
Example of using PyVRP solver to generate CVRPTW dataset.
"""
import numpy as np
from ml4co_kit import CVRPTWWrapper, CVRPTWGenerator, PyVRPSolver, CVRP_TYPE

# Set the wrapper for CVRPTW
cvrptw_wrapper = CVRPTWWrapper()

# Set the generator for CVRPTW
cvrptw_generator = CVRPTWGenerator(
    distribution_type=CVRP_TYPE.UNIFORM,
    precision=np.float32,
    nodes_num=50,
)

# Set the solver for CVRPTW
cvrptw_solver = PyVRPSolver(
    time_limit=1.0,
    seed=1234,
)

# Generate CVRPTW data
cvrptw_wrapper.generate_w_to_txt(
    file_path="cvrptw50_uniform.txt",
    generator=cvrptw_generator,
    solver=cvrptw_solver,
    num_samples=100,
    num_threads=4,
    show_time=True,
)
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository, license, and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/pyvrp](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/pyvrp).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `PyVRP` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
