# HGS-CVRP Integration in ML4CO-Kit

Dear HGS-CVRP Team,

We would like to sincerely thank you for developing and open-sourcing [`HGS-CVRP`](https://github.com/vidalt/HGS-CVRP).

`HGS-CVRP` is a modern implementation of the Hybrid Genetic Search (HGS) with advanced diversity control, specialized to the Capacitated Vehicle Routing Problem (CVRP). It also introduces the SWAP* neighborhood and has become a widely adopted high-performance heuristic for CVRP benchmarking and research. The open-source release has significantly improved the accessibility and reproducibility of SOTA CVRP heuristics within modern research ecosystems.

We have recently integrated `HGS-CVRP` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `HGS-CVRP` is currently integrated into the **solver** component under the **routing** category. The integration converts task instances to CVRPLIB format, invokes the bundled HGS executable, and parses the returned routes. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `HGS-CVRP` within `ML4CO-Kit` is as follows:

|  Solver   |  Task  | Status |                 Code                 |
|:---------:|:------:|:------:|:------------------------------------:|
| HGSSolver |  CVRP  | Supported | [ml4co_kit/solver/routing/lib/hgs/cvrp_hgs.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/routing/lib/hgs/cvrp_hgs.py) |

Below are simple examples of how `HGSSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using HGS solver to generate CVRP dataset.
"""
import numpy as np
from ml4co_kit import CVRPWrapper, CVRPGenerator, HGSSolver, CVRP_TYPE

# Set the wrapper for CVRP
cvrp_wrapper = CVRPWrapper()

# Set the generator for CVRP
cvrp_generator = CVRPGenerator(
    distribution_type=CVRP_TYPE.UNIFORM,
    precision=np.float32,
    nodes_num=50,
    min_demand=1,
    max_demand=9,
    min_capacity=40,
    max_capacity=40,
)

# Set the solver for CVRP
cvrp_solver = HGSSolver(hgs_time_limit=1.0)

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
Example of using HGS solver to solve CVRP dataset.
"""
from ml4co_kit import CVRPWrapper, HGSSolver

# Set the wrapper for CVRP
cvrp_wrapper = CVRPWrapper()

# Set the solver for CVRP
cvrp_solver = HGSSolver(hgs_time_limit=1.0)

# Read the dataset from txt file
cvrp_wrapper.from_txt(file_path="cvrp50_uniform_16ins.txt", ref=True)

# Solve the dataset using HGS solver
cvrp_wrapper.solve(solver=cvrp_solver, num_threads=4, show_time=True)

# Evaluate the solution
print(cvrp_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository, license, and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/hgs](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/hgs).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `HGS-CVRP` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
