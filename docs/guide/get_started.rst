=================================
Get Started
=================================


Basic Install by pip
----------------------

You can install the stable release on PyPI:

::

    $ pip install ml4co-kit

or get the latest version by running:

::

    $ pip install -U https://github.com/Thinklab-SJTU/ML4CO-Kit/archive/master.zip # with --user for user install (no root)


The following packages are required, and shall be automatically installed by ``pip``:

::

    Python >= 3.8
    numpy >= 1.24.4
    networkx >= 2.8.8
    tqdm >= 4.66.1
    pulp >= 2.8.0, 
    pandas >= 2.0.0,
    scipy >= 1.10.1
    aiohttp >= 3.9.3
    requests >= 2.31.0
    async_timeout >= 4.0.3
    pyvrp >= 0.6.3
    cython >= 3.0.8
    gurobipy >= 11.0.3


Example of ML4CO-Kit
-----------------------------------

Algorithm
>>>>>>>>>

.. code-block:: python
    :linenos:

    >>> from ml4co-kit import TSPSolver, tsp_insertion_decoder

    # create solver and load data
    >>> solver = TSPSolver()
    >>> solver.from_txt("your/txt/data/path", ref=True)

    # use insertion algorithm to solve the problems
    >>> tours = tsp_insertion_decoder(points=solver.points)
    >>> solver.from_data(tours=tours, ref=False)

    # evaluate (average length, ground truth, gap, std)
    >>> solver.evaluate(calculate_gap=True)
    (6.299320133465173, 5.790133693543183, 8.816004478345556, 3.605743337834312)


Draw
>>>>>>>>>

.. code-block:: python
    :linenos:
    
    >>> from ml4co_kit import mcl_solver
    >>> from ml4co-kit import draw_mcl_problem, draw_mcl_solution

    # use MClSolver to load the data
    >>> mcl_solver = MClSolver()
    >>> mcl_solver.from_txt("your/txt/data/path", ref=False)

    # draw problem and solution
    >>> draw_mcl_problem(
            graph_data=mcl_solver.graph_data[0],
            save_path="problem/image/save/path",
            self_loop=False 
        )
    >>> draw_mcl_solution(
            graph_data=mcl_solver.graph_data[0],
            save_path="solution/image/save/path", 
            self_loop=False 
        )


Evaluate
>>>>>>>>

.. code-block:: python
    :linenos:

    >>> from ml4co_kit.evaluate import TSPLIBOriEvaluator
    >>> from ml4co_kit.solver import TSPLKHSolver

    # create LKH solver
    >>> lkh_solver = TSPLKHSolver(scale=1)
    
    # create evaluator of TSPLIB
    >>> evaluator = TSPLIBOriEvaluator()
    
    # evaluate the LKH solver on the TSPLIB dataset (norm is EUC_2D)
    >>> evaluator.evaluate(lkh_solver, norm="EUC_2D")
            solved_costs      ref_costs          gaps
    eil51        429.983312     429.983312  0.000000e+00
    berlin52    7544.365902    7544.365902  3.616585e-14
    st70         677.881928     678.597452 -1.054416e-01
    eil76        544.837795     545.387552 -1.008012e-01
    pr76      108159.438274  108159.438274 -1.345413e-14
    kroA100    21285.443182   21285.443182  0.000000e+00
    kroC100    20750.762504   20750.762504  0.000000e+00
    kroD100    21294.290821   21294.290821  3.416858e-14
    rd100       7910.396210    7910.396210  0.000000e+00
    eil101       642.244814     642.309536 -1.007642e-02
    lin105     14382.995933   14382.995933  0.000000e+00
    ch130       6110.739012    6110.860950 -1.995428e-03
    ch150       6532.280933    6532.280933 -2.784616e-14
    tsp225      3859.000000    3859.000000  0.000000e+00
    a280        2587.930486    2586.769648  4.487600e-02
    pr1002    259066.663053  259066.663053  0.000000e+00
    pr2392    378062.826191  378062.826191  0.000000e+00
    AVG        50578.945903   50578.963027 -1.020227e-02

    # evaluate the LKH solver on the TSPLIB dataset (norm is GEO)
    >>> evaluator.evaluate(lkh_solver, norm="GEO")
            solved_costs    ref_costs          gaps
    ulysses16     74.108736    74.108736  1.917568e-14
    ulysses22     75.665149    75.665149  3.756248e-14
    gr96         512.309380   512.309380  0.000000e+00
    gr202        549.998070   549.998070 -8.268163e-14
    gr666       3843.137961  3952.535702 -2.767786e+00
    AVG         1011.043859  1032.923407 -5.535573e-01

Generator
>>>>>>>>

.. code-block:: python
    :linenos:

    >>> from ml4co_kit import TSPDataGenerator

    # create generator of TSP
    >>> tsp_data_lkh = TSPDataGenerator(
        num_threads=8,
        nodes_num=50,
        data_type="uniform",
        solver="LKH",
        train_samples_num=16,
        val_samples_num=16,
        test_samples_num=16,
        save_path="path/to/save/"
    )

    # generate
    tsp_data_lkh.generate()


What's Next
------------
Please see :doc:`../api/ml4co_kit` for the API documentation.