TaskBase
========

.. currentmodule:: ml4co_kit.task.base

:class:`TaskBase` is the root class for every problem type in ML4CO-Kit (routing, graph,
SAT, portfolio, etc.). It defines the common lifecycle:

1. **Load / construct** instance data (``from_pickle``, ``from_data`` in subclasses).
2. **Evaluate** a solution with :meth:`TaskBase.evaluate`.
3. **Compare** against a reference with :meth:`TaskBase.evaluate_w_gap`.
4. **Visualize** with :meth:`TaskBase.render` (implemented in subclasses).

Core attributes
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``task_type``
     - Problem identifier (:class:`TASK_TYPE`).
   * - ``minimize``
     - ``True`` if the objective is minimized (e.g. route length); ``False`` if maximized.
   * - ``precision``
     - Floating dtype, ``np.float32`` or ``np.float64``.
   * - ``sol``
     - Current solution encoding (problem-specific).
   * - ``ref_sol``
     - Reference / ground-truth solution for benchmarking.
   * - ``name``
     - Unique instance id (UUID hex by default).

Quick example
-------------

Load a CVRP task from pickle and evaluate the reference tour
(``cp311_base`` environment, ``test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl``):

.. code-block:: python

   import pathlib
   from ml4co_kit import CVRPTask

   task = CVRPTask()
   task.from_pickle(pathlib.Path(
       "test_dataset/routing/vrp/cvrp/task/cvrp50_uniform_task.pkl"
   ))
   print(task)
   # CVRPTask(2fb389cdafdb4e79a94572f01edf0b95)

   cost = task.evaluate(task.ref_sol)
   print(cost)
   # 10.973381996154785

Serialization
-------------

- :meth:`TaskBase.from_pickle` — restore a full task object from ``.pkl``.
- :meth:`TaskBase.to_pickle` — persist the task (used in tests for round-trip checks).
- :meth:`TaskBase.get_data_md5` — content hash for verifying data integrity
  (independent of pickle object identity).

.. code-block:: python

   print(task.get_data_md5())
   # 794ee6f7389c675300bddbe63453c4aa

Gap evaluation
--------------

:meth:`TaskBase.evaluate_w_gap` compares ``sol`` and ``ref_sol`` under the same
objective. For minimization problems the gap (%) is
``(sol_cost - ref_cost) / ref_cost * 100``.

.. code-block:: python

   task.sol = task.ref_sol.copy()
   sol_cost, ref_cost, gap = task.evaluate_w_gap()
   print(sol_cost, ref_cost, gap)
   # 10.973381996154785 10.973381996154785 0.0

API reference
-------------

.. autoclass:: TASK_TYPE
   :members:
   :undoc-members:

.. autoclass:: TaskBase
   :members:
   :inherited-members:
   :show-inheritance:
