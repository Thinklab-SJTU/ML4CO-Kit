import numpy as np
from ml4co_kit.algorithm.mis.c_arw import c_mis_two_improve


def mis_2improve_local_search(
    initial_sol: np.ndarray, xadj: np.ndarray, adjncy: np.ndarray,
) -> np.ndarray:
    # Number of nodes in the graph
    nodes_num = xadj.shape[0] - 1

    # Ensure input arrays are contiguous and of the correct type for the C extension
    xadj = np.ascontiguousarray(xadj, dtype=np.int32)
    initial_sol = np.ascontiguousarray(initial_sol, dtype=np.int32)
    adjncy = np.ascontiguousarray(adjncy, dtype=np.int32)

    # Allocate output array for the updated solution
    output = np.empty(nodes_num, dtype=np.int32)

    # Call the C extension function that performs the two-improvement local search
    c_mis_two_improve(nodes_num, xadj, adjncy, initial_sol, output)

    return output
