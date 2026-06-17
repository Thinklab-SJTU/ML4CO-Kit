CVRPTask
========

.. currentmodule:: ml4co_kit.task.routing.vrp.cvrp

The **Capacitated Vehicle Routing Problem (CVRP)** asks for a minimum-cost set of
routes from a depot that visits every customer exactly once, such that the total
demand on each route does not exceed vehicle capacity.

:class:`CVRPTask` is a single-instance container. For batch I/O, generation, and
parallel solving, use :class:`~ml4co_kit.wrapper.routing.vrp.cvrp.CVRPWrapper`.

Problem data
------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Shape / type
   * - ``depots``
     - ``(2,)`` or ``(3,)`` — depot coordinates (index ``0`` in ``coords``).
   * - ``points``
     - ``(V, 2)`` or ``(V, 3)`` — customer coordinates.
   * - ``coords``
     - ``(V+1, 2)`` — ``[depot; points]``, built automatically.
   * - ``demands``
     - ``(V,)`` — customer demands (depot demand is 0).
   * - ``capacity``
     - ``float`` — vehicle capacity.
   * - ``norm_demands``
     - ``(V,)`` — ``demands / capacity``.
   * - ``cvrp_open``
     - ``bool`` — if ``True``, routes need not return to depot (OVRP).
   * - ``sol`` / ``ref_sol``
     - ``1D`` tour with ``0`` as depot separators, e.g. ``[0,3,5,0,2,1,0]``.

Solution encoding
-----------------

Tours are stored as a **single 1D array**. Depot node index is ``0``; customer
indices are ``1 … V``. Multiple routes are concatenated with ``0`` delimiters:

.. code-block:: text

   [0,  customer_a, customer_b, 0,  customer_c, 0]
        └── route 1 ──────────┘    └ route 2 ┘

Constraints checked by :meth:`CVRPTask.check_constraints`:

- Tour starts and ends at depot ``0``.
- Every customer ``1…V`` appears exactly once.
- Per-route total demand ≤ ``capacity`` (with ``threshold`` tolerance).

Example 1 — Load from pickle
----------------------------

.. code-block:: python

   import pathlib
   from ml4co_kit import CVRPTask

   task = CVRPTask()
   task.from_pickle(pathlib.Path(
       "test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl"
   ))

   print(repr(task))
   # CVRPTask(2fb389cdafdb4e79a94572f01edf0b95)
   print(task.nodes_num, task.capacity)
   # 50 40.0

   cost = task.evaluate(task.ref_sol)
   print(cost)
   # 10.973381996154785

Example 2 — Build from raw arrays
---------------------------------

.. code-block:: python

   import numpy as np
   from ml4co_kit import CVRPTask

   task = CVRPTask()
   task.from_data(
       depots=np.array([0.0, 0.0], dtype=np.float32),
       points=np.array(
           [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
       ),
       demands=np.array([1.0, 1.0, 1.0], dtype=np.float32),
       capacity=2.0,
   )
   print(task.nodes_num)       # 3
   print(task.norm_demands)    # [0.5 0.5 0.5]

   # Two routes: {1,2} and {3}, capacity 2 each
   sol = np.array([0, 1, 2, 0, 3, 0])
   print(task.check_constraints(sol))  # True
   print(task.evaluate(sol))           # 6.242641

   # Single route with demand 3 > capacity 2
   bad = np.array([0, 1, 2, 3, 0])
   print(task.check_constraints(bad))  # False

Example 3 — VRPLIB I/O
----------------------

.. code-block:: python

   import pathlib
   from ml4co_kit import CVRPTask

   task = CVRPTask()
   task.from_vrplib(
       vrp_file_path=pathlib.Path(
           "test_dataset/routing/vrp/cvrp/vrplib/problem_1/X-n101-k25.vrp"
       ),
       sol_file_path=pathlib.Path(
           "test_dataset/routing/vrp/cvrp/vrplib/solution_1/X-n101-k25.sol"
       ),
       ref=True,
   )
   cost = task.evaluate(task.ref_sol)
   print(cost)

Example 4 — Visualization
-------------------------

.. code-block:: python

   import pathlib
   from ml4co_kit import CVRPTask

   task = CVRPTask()
   task.from_pickle(pathlib.Path(
       "test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl"
   ))
   task.sol = task.ref_sol
   task.render(
       save_path=pathlib.Path("docs/assets/cvrp_solution.png"),
       with_sol=True,
       figsize=(10, 10),
   )

See also :doc:`../../../../example` Case-03 for a full visualization walkthrough.

API reference
-------------

.. autoclass:: CVRPTask
   :members:
   :inherited-members:
   :show-inheritance:
