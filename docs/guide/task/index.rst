Task Module
===========

The **Task** module is the data layer of ML4CO-Kit. A :class:`~ml4co_kit.task.base.TaskBase`
instance holds a single combinatorial optimization problem: instance data, a candidate
solution (``sol``), and an optional reference solution (``ref_sol``).

Inheritance for routing problems:

.. code-block:: text

   TaskBase
     └── RoutingTaskBase
           └── CVRPTask  (and other VRP / TSP variants)

.. toctree::
   :maxdepth: 2

   base
   routing_base
   routing/vrp/cvrp
