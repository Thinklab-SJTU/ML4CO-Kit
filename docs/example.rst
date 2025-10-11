=================================
How to use ML4CO-Kit
=================================


Case-01: How to use ML4CO-Kit to generate a dataset
>>>>>>>>>>>>>

.. code-block:: python
    :linenos:

    # We take the TSP as an example

    # Import the required classes.
    >>> import numpy as np                  # Numpy
    >>> from ml4co_kit import TSPWrapper    # The wrapper for TSP, used to manage data and parallel generation.
    >>> from ml4co_kit import TSPGenerator  # The generator for TSP, used to generate a single instance.
    >>> from ml4co_kit import TSP_TYPE      # The distribution types supported by the generator.
    >>> from ml4co_kit import LKHSolver     # We choose LKHSolver to solve TSP instances

    # Check which distributions are supported by the TSP types.
    >>> for type in TSP_TYPE:
    ...     print(type)
    TSP_TYPE.UNIFORM
    TSP_TYPE.GAUSSIAN
    TSP_TYPE.CLUSTER

    # Set the generator parameters according to the requirements.
    >>> tsp_generator = TSPGenerator(
    ...     distribution_type=TSP_TYPE.GAUSSIAN,   # Generate a TSP instance with a Gaussian distribution
    ...     precision=np.float32,                  # Floating-point precision: 32-bit
    ...     nodes_num=50,                          # Number of nodes in TSP instance
    ...     gaussian_mean_x=0,                     # Mean of Gaussian for x coordinate
    ...     gaussian_mean_y=0,                     # Mean of Gaussian for y coordinate
    ...     gaussian_std=1,                        # Standard deviation of Gaussian
    ... )

    # Set the LKH parameters.
    >>> tsp_solver = LKHSolver(
    ...     lkh_scale=1e6,        # Scaling factor to convert floating-point numbers to integers
    ...     lkh_max_trials=500,   # Maximum number of trials for the LKH algorithm
    ...     lkh_path="LKH",       # Path to the LKH executable
    ...     lkh_runs=1,           # Number of runs for the LKH algorithm
    ...     lkh_seed=1234,        # Random seed for the LKH algorithm
    ...     lkh_special=False,    # When set to True, disables 2-opt and 3-opt heuristics
    ... )

    # Create the TSP wrapper
    >>> tsp_wrapper = TSPWrapper(precision=np.float32)

    # Use ``generate_w_to_txt`` to generate a dataset of TSP.
    >>> tsp_wrapper.generate_w_to_txt(
    ...     file_path="tsp_gaussian_16ins.txt",  # Path to the output file where the generated TSP instances will be saved
    ...     generator=tsp_generator,             # The TSP instance generator to use
    ...     solver=tsp_solver,                   # The TSP solver to use
    ...     num_samples=16,                      # Number of TSP instances to generate
    ...     num_threads=4,                       # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
    ...     batch_size=1,                        # Batch size for parallel processing; cannot both be non-1 with num_threads
    ...     write_per_iters=1,                   # Number of sub-generation steps after which data will be written to the file
    ...     write_mode="a",                      # Write mode for the output file ("a" for append)
    ...     show_time=True,                      # Whether to display the time taken for the generation process
    ... )
    Generating TSP: 100%|..........| 4/4 [00:00<00:00, 12.79it/s]


Case-02: How to use ML4CO-Kit to load problems and solve them
>>>>>>>>>>>>>

.. code-block:: python
    :linenos:

    # We take the MIS as an example

    # Import the required classes.
    >>> import numpy as np                  # Numpy
    >>> from ml4co_kit import MISWrapper    # The wrapper for MIS, used to manage data and parallel solving.
    >>> from ml4co_kit import KaMISSolver   # We choose KaMISSolver to solve MIS instances

    # Set the KaMIS parameters.
    >>> mis_solver = KaMISSolver(
    ...     kamis_time_limit=10.0,          # The maximum solution time for a single problem
    ...     kamis_weighted_scale=1e5,       # Weight scaling factor, used when nodes have weights.
    ... )

    # Create the MIS wrapper
    >>> mis_wrapper = MISWrapper(precision=np.float32)

    # Load the problems to be solved.
    # You can use the corresponding loading function based on the file type, 
    # such as ``from_txt`` for txt file and ``from_pickle`` for pickle file.
    >>> mis_wrapper.from_txt(
    ...     file_path="test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.txt",
    ...     ref=True,          # TXT file contains labels. Set ``ref=True`` to set them as reference.
    ...     overwrite=True,    # Whether to overwrite the data. If not, only update according to the file data.
    ...     show_time=True     # Whether to display the time taken for the loading process
    ... )
    Loading data from test_dataset/mis/wrapper/mis_rb-small_uniform-weighted_4ins.txt: 4it [00:00, 75.41it/s]

    # Use ``solve`` to call the KaMISSolver to perform the solution.
    >>> mis_wrapper.solve(
    ...     solver=mis_solver,                   # The solver to use
    ...     num_threads=2,                       # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
    ...     batch_size=1,                        # Batch size for parallel processing; cannot both be non-1 with num_threads
    ...     show_time=True,                      # Whether to display the time taken for the generation process
    ... )
    Solving MIS Using kamis: 100%|..........| 2/2 [00:21<00:00, 10.97s/it]
    Using Time: 21.947036743164062

    # Use ``evaluate_w_gap`` to obtain the evaluation results.
    # Evaluation Results: average solution value, average reference value, gap (%), gap std.
    >>> eval_result = mis_wrapper.evaluate_w_gap()
    >>> print(eval_result)
    (14.827162742614746, 15.18349838256836, 2.5054726600646973, 2.5342845916748047)


Case-03: How to use ML4CO-Kit to visualize the COPs
>>>>>>>>>>>>>

.. code-block:: python
    :linenos:

    # We take the CVRP as an example

    # Import the required classes.
    >>> import numpy as np                  # Numpy
    >>> from ml4co_kit import CVRPTask      # CVRP Task. 
    >>> from ml4co_kit import CVRPWrapper   # The wrapper for CVRP, used to manage data.

    # Case-1: multiple task data are saved in ``txt``, ``pickle``, etc. single task data is saved in pickle.
    >>> cvrp_wrapper = CVRPWrapper()
    >>> cvrp_wrapper.from_pickle("test_dataset/cvrp/wrapper/cvrp50_uniform_16ins.pkl")
    >>> cvrp_task = cvrp_wrapper.task_list[0]
    >>> print(cvrp_task)
    CVRPTask(2fb389cdafdb4e79a94572f01edf0b95)

    # Case-2: single task data is saved in pickle.
    >>> cvrp_task = CVRPTask()
    >>> cvrp_task.from_pickle("test_dataset/cvrp/task/cvrp50_uniform_task.pkl")
    >>> print(cvrp_task)
    CVRPTask(2fb389cdafdb4e79a94572f01edf0b95)

    # The loaded solution is usually a reference solution. 
    # When drawing the image, it is the ``sol`` that is being drawn. 
    # Therefore, it is necessary to assign ``ref_sol`` to ``sol``.
    >>> cvrp_task.sol = cvrp_task.ref_sol

    # Using ``render`` to get the visualization
    >>> cvrp_task.render(
    ...     save_path="./docs/assets/cvrp_solution.png",  # Path to save the rendered image
    ...     with_sol=True,                                # Whether to draw the solution tour
    ...     figsize=(10, 10),                             # Size of the image (width and height)
    ...     node_color="darkblue",                        # Color of the nodes
    ...     edge_color="darkblue",                        # Color of the edges
    ...     node_size=50                                  # Size of the nodes
    ... )


Case-04: A simple ML4CO example
>>>>>>>>>>>>>

.. code-block:: python
    :linenos:

    # We take the MCut as an example

    # Import the required classes.
    >>> import numpy as np                   # Numpy
    >>> from ml4co_kit import MCutWrapper    # The wrapper for MCutWrapper, used to manage data.
    >>> from ml4co_kit import GreedySolver   # GreedySolver, based on GNN4CO.
    >>> from ml4co_kit import RLSAOptimizer  # Using RLSA to perform local search.
    >>> from ml4co_kit.extension.gnn4co import GNN4COModel, GNN4COEnv, GNNEncoder

    # Set the GNN4COModel parameters. ``weight_path``: Pretrain weight path. 
    # If it is not available locally, it will be automatically downloaded from Hugging Face.
    >>> gnn4mcut_model = GNN4COModel(
    ...     env=GNN4COEnv(
    ...         task="MCut",              # Task name: MCut.                                 
    ...         mode="solve",             # Mode: solving mode.
    ...         sparse_factor=1,          # Sparse factor: Controls the sparsity of the graph.
    ...         device="cuda"             # Device: 'cuda' or 'cpu'
    ...     ),
    ...     encoder=GNNEncoder(
    ...         task="MCut",              # Task name: MCut.
    ...         sparse=True,              # Graph data should set ``sparse`` to True.
    ...         block_layers=[2,4,4,2]    # Block layers: the number of layers in each block of the encoder.
    ...     ),
    ...     weight_path="weights/gnn4co_mcut_ba-large_sparse.pt"   
    ... )
    gnn4co/gnn4co_mcut_ba-large_sparse.pt: 100% |..........| 19.6M/19.6M [00:03<00:00, 6.18MB/s]

    # Set the RLSAOptimizer parameters.
    >>> mcut_optimizer = RLSAOptimizer(
    ...     rlsa_kth_dim="both",          # Which dimension to consider for the k-th value calculation.
    ...     rlsa_tau=0.01,                # The temperature parameter in the Simulated Annealing process.
    ...     rlsa_d=2,                     # Control the step size of each update.
    ...     rlsa_k=1000,                  # The number of samples used in the optimization process.
    ...     rlsa_t=1000,                  # The number of iterations in the optimization process.
    ...     rlsa_device="cuda",           # Device: 'cuda' or 'cpu'.
    ...     rlsa_seed=1234                # The random seed for reproducibility.
    ... )

    # Set the GreedySolver parameters.
    >>> mcut_solver_wo_opt = GreedySolver(
    ...     model=gnn4mcut_model,         # GNN4CO model for MCut
    ...     device="cuda",                # Device: 'cuda' or 'cpu'.
    ...     optimizer=None                # The optimizer to perform local search.
    ... )
    >>> mcut_solver_w_opt = GreedySolver(
    ...     model=gnn4mcut_model,         # GNN4CO model for MCut
    ...     device="cuda",                # Device: 'cuda' or 'cpu'.
    ...     optimizer=mcut_optimizer      # The optimizer to perform local search.
    ... )

    # Create the MCut wrapper
    >>> mcut_wrapper = MCutWrapper(precision=np.float32)

    # Load the problems to be solved.
    # You can use the corresponding loading function based on the file type, 
    # such as ``from_txt`` for txt file and ``from_pickle`` for pickle file.
    >>> mcut_wrapper.from_txt(
    ...     file_path="test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.txt",
    ...     ref=True,          # TXT file contains labels. Set ``ref=True`` to set them as reference.
    ...     overwrite=True,    # Whether to overwrite the data. If not, only update according to the file data.
    ...     show_time=True     # Whether to display the time taken for the loading process
    ... )
    Loading data from test_dataset/mcut/wrapper/mcut_ba-large_no-weighted_4ins.txt: 4it [00:00, 16.35it/s]

    # Using ``solve`` to get the solution (without optimizer)
    >>> mcut_wrapper.solve(
    ...     solver=mcut_solver_wo_opt,    # The solver to use
    ...     num_threads=1,                # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
    ...     batch_size=1,                 # Batch size for parallel processing; cannot both be non-1 with num_threads
    ...     show_time=True,               # Whether to display the time taken for the generation process
    ... )
    Solving MCut Using greedy: 100%|..........| 4/4 [00:00<00:00, 12.34it/s]
    Using Time: 0.3261079788208008

    # Use ``evaluate_w_gap`` to obtain the evaluation results.
    # Evaluation Results: average solution value, average reference value, gap (%), gap std.
    >>> eval_result = mcut_wrapper.evaluate_w_gap()
    >>> print(eval_result)
    (2647.25, 2726.5, 2.838811523236064, 0.7528157058230817)

    # Using ``solve`` to get the solution (with optimizer)
    >>> mcut_wrapper.solve(
    ...     solver=mcut_solver_w_opt,     # The solver to use
    ...     num_threads=1,                # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
    ...     batch_size=1,                 # Batch size for parallel processing; cannot both be non-1 with num_threads
    ...     show_time=True,               # Whether to display the time taken for the generation process
    ... )
    Solving MCut Using greedy: 100%|..........| 4/4 [00:02<00:00,  1.46it/s]
    Using Time: 2.738525867462158

    # Use ``evaluate_w_gap`` to obtain the evaluation results.
    # Evaluation Results: average solution value, average reference value, gap (%), gap std.
    >>> eval_result = mcut_wrapper.evaluate_w_gap()
    >>> print(eval_result)
    (2693.0, 2726.5, 1.2373146256952277, 0.29320238806274546)


What's Next
-----------
Please see :doc:`../api/ml4co_kit` for the API documentation.