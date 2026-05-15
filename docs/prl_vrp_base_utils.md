# PRL VRP Base Utilities

## Purpose

This PR adds the first layer of PRL infrastructure for vehicle routing tasks in ML4CO-Kit. It focuses on generic depot-delimited VRP solution utilities and thin CVRP-facing helper methods.

PRL, Prior-guided Reinforcement Learning, conditions routing decisions on a reference solution prior. For routing tasks, a reference solution can be represented as directed edges, a node-to-successor heatmap, and node-level confidence values. These utilities prepare that representation without changing solver, optimizer, or learning code.

## Supported Solution Format

The current standard format is a depot-delimited flat solution:

```text
[0, 3, 5, 0, 2, 1, 0]
```

This represents two CVRP routes:

```text
0 -> 3 -> 5 -> 0
0 -> 2 -> 1 -> 0
```

The standard format should start and end with depot `0`. The splitting utility skips empty routes caused by consecutive depots and handles missing boundary depots for compatibility.

## Implemented

The new `ml4co_kit.task.routing.vrp_base` module provides:

- `split_routes`
- `routes_to_flat`
- `sol_to_edges`
- `sol_to_heatmap`
- `sol_to_successor`
- `check_visit_once`
- `get_route_demands`
- `get_capacity_violation`
- `check_capacity`
- `evaluate_distance`
- `make_prior`

`CVRPTask` exposes thin wrappers for the same solution and prior utilities while preserving the existing CVRP data format. In particular, `CVRPTask.demands` still stores customer demands only, so the wrapper inserts depot demand internally when calling the generic capacity utilities.

## Depot Successor Handling

In CVRP, the depot appears once per route boundary and can have multiple successors. A plain successor vector cannot represent all depot outgoing edges.

The successor utility therefore uses this convention:

- `successor[i] = j` for customer nodes.
- `successor[0] = -1` by default.
- `depot_successors` stores all depot successors, for example `[3, 2]`.

The heatmap keeps all directed depot edges, so multiple `heatmap[0, j] = 1` entries are valid.

## Prior Format

`make_prior` returns:

```python
{
    "heatmap": heatmap,
    "edge_list": edge_list,
    "successor": successor,
    "depot_successors": depot_successors,
    "confidence": confidence,
}
```

The current confidence support is intentionally simple:

- `confidence=None` creates uniform node confidence with value `1.0`.
- A scalar confidence is broadcast to all nodes.
- A vector confidence must have shape `(num_nodes,)`.

## Out of Scope

This PR does not:

- Train PRL.
- Implement PyVRP solver support.
- Add VRPL, VRPTW, or OVRP task files.
- Implement OR-Tools, LKH, HGS, or local search logic.
- Modify neural models, decoders, QKV computation, attention bias, FiLM, gates, or Bernoulli priors.
- Reproduce MVMoE, PIP, DACT, or NeuOpt model code.
- Add new heavy dependencies.

## Next PRs

- PR-2: PyVRPSolver for CVRP.
- PR-3: VRPL.
- PR-4: VRPTW.
- PR-5: OVRP.
- PR-6: Lightweight local search optimizer.

