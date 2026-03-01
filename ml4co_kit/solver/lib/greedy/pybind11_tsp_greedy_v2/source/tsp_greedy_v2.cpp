#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "tsp_greedy.hpp"
#include "parallel.hpp"

namespace py = pybind11;

// Type alias for 32-bit integer
using int32 = int32_t;

/**
 * @brief Greedy insertion algorithm for TSP using PyBind11
 * 
 * Constructs TSP tour(s) from candidate edges using greedy insertion.
 * Supports both single instance and batched processing with parallel execution.
 * 
 * @param candidate_edges Edge array(s):
 *                        - Single instance: 1D array of shape (E,) where E is even
 *                          Each pair (candidate_edges[2*k], candidate_edges[2*k+1]) represents an edge
 *                        - Batched: 2D array of shape (batch_size, E)
 * @param num_nodes Number of nodes in the TSP instance
 * @param num_workers Number of worker threads for parallel execution (for batched mode)
 * 
 * @return Tour(s):
 *         - Single instance: 1D array of shape (num_nodes + 1,)
 *         - Batched: 2D array of shape (batch_size, num_nodes + 1)
 */
inline py::array_t<int> tsp_greedy_insert(
    const py::array_t<int> & candidate_edges,
    const int num_nodes,
    const int num_workers
) {
    // Ensure array is C-style contiguous
    assert(candidate_edges.flags() & py::array::c_style);

    const bool batched = candidate_edges.ndim() == 2;

    if (!batched) {
        // Single instance processing
        const auto edges_ptr = candidate_edges.data();
        const int num_candidate_edges = candidate_edges.shape()[0] / 2;

        // Allocate output tour array
        py::array_t<int> tour({num_nodes + 1});
        auto tour_ptr = static_cast<int *>(tour.request().ptr);

        // Release GIL for better performance during computation
        pybind11::gil_scoped_release release;

        // Perform greedy insertion
        tsp::greedy_insert(tour_ptr, edges_ptr, num_nodes, num_candidate_edges);

        return tour;
    } else {
        // Batched processing
        const int batch_size = candidate_edges.shape()[0];
        const int num_candidate_edges = candidate_edges.shape()[1] / 2;
        const auto edges_ptr = candidate_edges.data();

        // Allocate output tours array
        py::array_t<int> tours({batch_size, num_nodes + 1});
        auto tours_ptr = static_cast<int *>(tours.request().ptr);

        // Release GIL for better performance during computation
        pybind11::gil_scoped_release release;

        // Define task function for parallel execution
        auto task_fn = [&](const int task_id) {
            // Process tour for this task
            tsp::greedy_insert(
                tours_ptr + task_id * (num_nodes + 1),
                edges_ptr + task_id * num_candidate_edges * 2,
                num_nodes,
                num_candidate_edges
            );
        };

        // Execute tasks in parallel
        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

        return tours;
    }
}

/**
 * @brief PyBind11 module for TSP greedy insertion
 * 
 * Exposes the tsp_greedy_insert function to Python.
 */
PYBIND11_MODULE(tsp_greedy_v2_impl, m) {
    m.doc() = "PyBind11 implementation of greedy insertion for TSP.";
    
    m.def(
        "tsp_greedy_insert",
        &tsp_greedy_insert,
        py::arg("candidate_edges"),
        py::arg("num_nodes"),
        py::arg("num_workers"),
        "Construct TSP tour(s) from candidate edges using greedy insertion.\n"
        "\n"
        "Parameters:\n"
        "  candidate_edges: Edge list as 1D array (E,) or 2D array (B, E)\n"
        "                   where E is even and each pair represents an edge (i, j)\n"
        "  num_nodes: Number of nodes in the TSP instance\n"
        "  num_workers: Number of worker threads for parallel execution\n"
        "\n"
        "Returns:\n"
        "  Tour(s) as 1D array (N+1,) or 2D array (B, N+1)"
    );
}

