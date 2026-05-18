# PyVRPSolver Notes

## Scope

`PyVRPSolver` currently supports CVRP only. It is a routing solver wrapper around PyVRP and follows ML4CO-Kit's existing `SolverBase.solve(task_data)` flow by writing the solution back to `task_data.sol`.

## Optional Dependency

PyVRP is imported lazily inside the CVRP backend. Importing ML4CO-Kit or `PyVRPSolver` should not import PyVRP immediately.

Install PyVRP before solving with this solver:

```bash
pip install pyvrp
```

If PyVRP is missing, solving raises an `ImportError` with an installation hint.

## Data Conversion

The solver reads the current `CVRPTask` fields:

- `depots`
- `points`
- `coords`
- `demands`
- `capacity`
- the Kit distance evaluator via `task_data._get_dists()`

ML4CO-Kit computes the full distance matrix first, then the backend scales it to integer distances for PyVRP:

- `pyvrp_distance_scale`
- `pyvrp_demand_scale`

Demand and capacity are also scaled to integer values before they are passed to PyVRP.

## Solution Format

The solution written back to `task_data.sol` remains the ML4CO-Kit flat depot-delimited CVRP format:

```text
[0, ..., 0, ..., 0]
```

For example:

```text
[0, 1, 2, 0, 3, 4, 0]
```

After solving, the backend checks `task_data.check_constraints(task_data.sol)` and recomputes the cost with the existing CVRP evaluator.

## Not Supported In This PR

- VRPL
- VRPTW
- OVRP
- local search optimizer
- PRL training
- neural model or decoder changes

## Next PRs

- PR-3: VRPL task/generator/wrapper
- PR-4: VRPTW task/generator/wrapper
- PR-5: OVRP task/generator/wrapper

