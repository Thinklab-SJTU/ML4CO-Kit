import numpy as np
from ml4co_kit.algorithm.mis.c_arw import c_mis_evo


def mis_evo_decoder(
    xadj: np.ndarray, adjncy: np.ndarray, time_limit: float = 1.0
) -> np.ndarray:
    """
    Perform one iteration of local search for the Maximum Independent Set (MIS)
    using the ARW (Andrade-Resende-Werneck) two-improvement algorithm.
    """
    # Number of nodes in the graph
    nodes_num = xadj.shape[0] - 1

    # Ensure input arrays are contiguous and of the correct type for the C extension
    xadj = np.ascontiguousarray(xadj, dtype=np.int32)
    adjncy = np.ascontiguousarray(adjncy, dtype=np.int32)

    # Allocate output array for the updated solution
    output = np.empty(nodes_num, dtype=np.int32)

    # Call the C extension function that performs the two-improvement local search
    c_mis_evo(nodes_num, xadj, adjncy, time_limit, output)

    return output
