RoutingTaskBase
===============

.. currentmodule:: ml4co_kit.task.routing.base

:class:`RoutingTaskBase` extends :class:`~ml4co_kit.task.base.TaskBase` for problems
defined on a set of nodes with pairwise travel costs (TSP, VRP, etc.).

Compared to :class:`TaskBase`, it adds:

- ``distance_type`` — how edge lengths are computed (:class:`DISTANCE_TYPE`).
- ``dist_eval`` — a :class:`DistanceEvaluator` used by subclasses for
  :meth:`DistanceEvaluator.cal_distance` and :meth:`DistanceEvaluator.cal_dist_matrix`.

Distance types
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``DISTANCE_TYPE``
     - Meaning
   * - ``EUC_2D`` / ``EUC_3D``
     - Euclidean distance (default for synthetic datasets).
   * - ``MAN_2D`` / ``MAN_3D``
     - Manhattan (L1) distance.
   * - ``MAX_2D`` / ``MAX_3D``
     - Chebyshev (L∞) distance.
   * - ``GEO``
     - Great-circle distance on a sphere (TSPLIB GEO format).
   * - ``ATT``
     - ATT pseudo-Euclidean metric (TSPLIB).

Rounding
--------

:class:`ROUND_TYPE` controls integer rounding of computed distances
(``NO``, ``CEIL``, ``FLOOR``, ``ROUND``), matching TSPLIB / VRPLIB conventions.

DistanceEvaluator
-----------------

:class:`DistanceEvaluator` centralizes metric logic so TSP and VRP tasks share the
same distance semantics:

.. code-block:: python

   import numpy as np
   from ml4co_kit.task.routing.base import (
       DistanceEvaluator, DISTANCE_TYPE, ROUND_TYPE,
   )

   dist_eval = DistanceEvaluator(
       distance_type=DISTANCE_TYPE.EUC_2D,
       round_type=ROUND_TYPE.NO,
   )
   a = np.array([0.0, 0.0])
   b = np.array([3.0, 4.0])
   print(dist_eval.cal_distance(a, b))
   # 5.0

   coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
   print(dist_eval.cal_dist_matrix(coords))
   # [[0. 1. 1.]
   #  [1. 0. 1.4142135]
   #  [1. 1.4142135 0. ]]

Subclass contract
-----------------

Concrete routing tasks (e.g. :class:`~ml4co_kit.task.routing.vrp.cvrp.CVRPTask`) must implement:

- :meth:`~ml4co_kit.task.base.TaskBase.from_data` — populate coordinates and problem fields.
- :meth:`~ml4co_kit.task.base.TaskBase.check_constraints` — feasibility check.
- :meth:`~ml4co_kit.task.base.TaskBase.evaluate` — tour / route cost.
- :meth:`~ml4co_kit.task.base.TaskBase.render` — matplotlib visualization.

API reference
-------------

.. autoclass:: DISTANCE_TYPE
   :members:
   :undoc-members:

.. autoclass:: ROUND_TYPE
   :members:
   :undoc-members:

.. autoclass:: DistanceEvaluator
   :members:
   :show-inheritance:

.. autoclass:: RoutingTaskBase
   :members:
   :inherited-members:
   :show-inheritance:
