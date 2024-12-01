=================================
Introduction and Guidelines
=================================

This page provides a brief introduction to make you understand how machine learning 
practices on combinatorial optimization (CO) problems and some guidelines for using ``ml4co-kit``.

.. note::
    For more information on CO problems and technical details, please visit our curated 
    repository `awesome-ml4co <https://github.com/Thinklab-SJTU/awesome-ml4co>`_.


Combinatorial Optimization Problems
-----------------------------------

- **Traveling Salesman Problem (TSP).** The TSP problem requires finding the shortest tour that visits each vertex of the graph exactly once and returns to the starting node. 

.. image:: ../assets/tsp_problem.png
    :width: 150px
    :height: 150px
.. image:: ../assets/tsp_solution.png
    :width: 150px
    :height: 150px

- **Asymmetric Traveling Salesman Problem (ATSP).** The ATSP is a variation of the TSP where the travel costs between nodes (cities) are not symmetric. 

- **Capacitated Vehicle Routing Problem (CVRP).** The CVRP is a variant of the vehicle routing problem in which a fleet of vehicles must service a set of customers, subject to vehicle capacity constraints. The goal is to minimize the total route cost while ensuring that no vehicle exceeds its capacity.

.. image:: ../assets/cvrp_problem.png
    :width: 150px
    :height: 150px
.. image:: ../assets/cvrp_solution.png
    :width: 150px
    :height: 150px

- **Maximum Clique (MCl).** The MCl problem involves finding the largest subset of vertices in a graph such that every pair of vertices in the subset is connected by an edge.

.. image:: ../assets/mcl_problem.png
    :width: 150px
    :height: 150px
.. image:: ../assets/mcl_solution.png
    :width: 150px
    :height: 150px

- **Maximum Cut (MCut).**  The MCut problem focuses on partitioning the vertices of a graph into two subsets such that the total weight of edges between the two subsets is maximized.

.. image:: ../assets/mcut_problem.png
    :width: 150px
    :height: 150px
.. image:: ../assets/mcut_solution.png
    :width: 150px
    :height: 150px

- **Maximum Independent Set (MIS).** The MIS problem aims to find the largest subset of a graph such that no two vertices in the subset are adjacent.

.. image:: ../assets/mis_problem.png
    :width: 150px
    :height: 150px
.. image:: ../assets/mis_solution.png
    :width: 150px
    :height: 150px

- **Minimum Vertex Cover (MVC).** The MVC problem aims to find the smallest subset of a graph such that every edge in the graph has at least one endpoint in this subset.

.. image:: ../assets/mvc_problem.png
    :width: 150px
    :height: 150px
.. image:: ../assets/mvc_solution.png
    :width: 150px
    :height: 150px


When to use ML4CO-Kit
-----------------------------------

In the following situations, you might find the ML4CO-Kit useful:

- When you need to obtain a baseline by solving CO problems with traditional solvers.

- When you need to generate training datasets for your models.

- When you require evaluating your models with open-source or public test datasets.

- When you are building your models using the PyTorch Lightning architecture.

- When you utilize wandb for monitoring your training process.

- When you need visualization of CO problems.

- When you need to read data from various formats such as ``tsplib``, ``vrplib``, ``txt``, and ``gpickle``.

- When you need to save data in a portable ``txt`` file format.


What's Next
------------
Please read the :doc:`get_started` guide.