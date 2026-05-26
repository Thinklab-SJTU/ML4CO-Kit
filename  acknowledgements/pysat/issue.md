# PySAT Integration in ML4CO-Kit

Dear PySAT Team,

We would like to sincerely thank you for developing and open-sourcing [`PySAT`](https://github.com/pysathq/pysat).

`PySAT` is a Python toolkit for prototyping with SAT oracles, providing unified access to state-of-the-art SAT solvers and supporting a wide range of SAT-related research workflows. It has become a standard infrastructure for satisfiability testing and related combinatorial problems in the CO and ML research communities.

We have recently integrated `PySAT` into our open-source project [`ML4CO-Kit`](https://github.com/Thinklab-SJTU/ML4CO-Kit), which aims to provide foundational infrastructure support for machine learning practices on COPs.

---

## ML4CO-Kit Framework

`ML4CO-Kit` is designed as a unified infrastructure framework for COPs and machine learning research. Our current system is organized into five major levels as below. Within this architecture, `PySAT` is currently integrated into the **solver** component under the **SAT** category. The integration converts task instances into CNF representations, invokes PySAT-backed SAT oracles, and parses the returned satisfiability predictions or satisfying assignments. The integration is designed to support unified solver invocation, dataset generation, benchmarking, and downstream ML4CO research workflows.

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

At the current stage, the support status of `PySAT` within `ML4CO-Kit` is as follows:

|   Solver    |  Task  | Status |                 Code                 |
|:-----------:|:------:|:------:|:------------------------------------:|
| PySATSolver |  SATP  | Supported | [ml4co_kit/solver/sat/lib/pysat/satp_pysat.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/sat/lib/pysat/satp_pysat.py) |
| PySATSolver |  SATA  | Supported | [ml4co_kit/solver/sat/lib/pysat/sata_pysat.py](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/ml4co_kit/solver/sat/lib/pysat/sata_pysat.py) |

Below are simple examples of how `PySATSolver` can be used within `ML4CO-Kit`.

```python
r"""
Example of using PySAT solver to generate SATP dataset.
"""
from ml4co_kit import SATPWrapper, SATPGenerator, PySATSolver, SAT_TYPE

# Set the wrapper for SATP
satp_wrapper = SATPWrapper()

# Set the generator for SATP
satp_generator = SATPGenerator(
    distribution_type=SAT_TYPE.CA,
    ca_n_range=(40, 50),
)

# Set the solver for SATP (default backend: CaDiCaL)
satp_solver = PySATSolver(
    pysat_solver_name="cadical195",
)

# Generate SATP data
satp_wrapper.generate_w_to_txt(
    file_path="satp_ca-small.txt",
    generator=satp_generator,
    solver=satp_solver,
    num_samples=100,
    num_threads=4,
    show_time=True,
)
```

```python
r"""
Example of using PySAT solver to solve SATA dataset.
"""
from ml4co_kit import SATAWrapper, PySATSolver

# Set the wrapper for SATA
sata_wrapper = SATAWrapper()

# Set the solver for SATA
sata_solver = PySATSolver(pysat_solver_name="cadical195")

# Read the dataset from txt file
sata_wrapper.from_txt(file_path="sata_ca-small_16ins.txt", ref=True)

# Solve the dataset using PySAT solver
sata_wrapper.solve(solver=sata_solver, num_threads=4, show_time=True)

# Evaluate the solution
print(sata_wrapper.evaluate_w_gap())
```

---

## Attribution and Acknowledgement

We explicitly preserve attribution to the original repository, license, and associated papers throughout our documentation and acknowledgement system. Detailed acknowledgement and integration documentation can be found at [acknowledgements/pysat](https://github.com/Thinklab-SJTU/ML4CO-Kit/blob/main/acknowledgements/pysat).

We would also sincerely welcome any suggestions, updates, improvements, or further discussions regarding the integration and future development of `PySAT` within `ML4CO-Kit`. Please feel free to open an issue or contact us via `heatingma@sjtu.edu.cn`. We greatly appreciate your contribution to the CO community and look forward to potential future collaboration.

Best regards,
heatingma, ML4CO-Kit Team
