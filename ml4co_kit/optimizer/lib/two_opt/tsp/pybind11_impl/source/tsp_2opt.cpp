#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include "tsp.hpp"
#include "parallel.hpp"

namespace py = pybind11;

// Type alias for 32-bit integer
using int32 = int32_t;

/**
 * @brief Two-opt local search for TSP using PyBind11
 * 
 * Optimizes one or multiple TSP tours using the two-opt local search algorithm.
 * Supports both single tour and batched tour optimization with parallel execution.
 * 
 * @param tours Tour array(s) to optimize:
 *              - Single tour: 1D array of shape (num_nodes,)
 *              - Batched tours: 2D array of shape (batch_size, num_nodes)
 * @param dists Distance matrix/matrices:
 *              - Single: 2D array of shape (num_nodes, num_nodes)
 *              - Batched (N tours + N dists): 3D array of shape (batch_size, num_nodes, num_nodes)
 *              - Batched (N tours + 1 dist): 2D array of shape (num_nodes, num_nodes) - shared by all tours
 * @param num_steps Maximum number of two-opt iterations to perform
 * @param type_2opt Type of 2-opt search:
 *                  - 1: Return as soon as an improving pair is found
 *                  - 2: Find the best pair in each iteration
 * @param num_workers Number of worker threads for parallel execution (for batched mode)
 * 
 * @return Optimized tour(s) (same shape as input tours)
 */
inline py::array_t<int> two_opt_local_search(
    py::array_t<int> & tours, 
    const py::array_t<float> & dists,
    const int num_steps, 
    const int type_2opt, 
    const int num_workers
) {
    // Ensure arrays are C-style contiguous
    assert(tours.flags() & py::array::c_style);
    assert(dists.flags() & py::array::c_style);

    const bool batched = tours.ndim() == 2;

    if (!batched) {
        // Single tour optimization
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        const auto dists_ptr = dists.data();
        const int num_nodes = dists.shape()[0];

        // Release GIL for better performance during computation
        pybind11::gil_scoped_release release;

        // Perform two-opt local search
        tsp::two_opt(tours_ptr, dists_ptr, num_nodes, num_steps, type_2opt);
    } else {
        // Batched tour optimization
        const int batch_size = tours.shape()[0];
        auto tours_ptr = static_cast<int *>(tours.request().ptr);
        const auto dists_ptr = dists.data();
        const int num_nodes = dists.shape()[1];
        const int tour_size = tours.shape()[1];  // Should be num_nodes + 1

        // Check if dists is shared (2D) or batched (3D)
        const bool shared_dists = dists.ndim() == 2;

        // Release GIL for better performance during computation
        pybind11::gil_scoped_release release;

        if (shared_dists) {
            // N tours + 1 dist: All tours share the same distance matrix
            auto task_fn = [&](const int task_id) {
                // Optimize tour for this task using the shared distance matrix
                tsp::two_opt(
                    tours_ptr + task_id * tour_size,
                    dists_ptr,  // Use the same distance matrix for all tours
                    num_nodes, 
                    num_steps, 
                    type_2opt
                );
            };

            // Execute tasks in parallel
            parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
        } else {
            // N tours + N dists: Each tour has its own distance matrix
            auto task_fn = [&](const int task_id) {
                // Optimize tour for this task
                // Note: tour_size = num_nodes + 1 (includes starting and ending node)
                tsp::two_opt(
                    tours_ptr + task_id * tour_size,
                    dists_ptr + task_id * num_nodes * num_nodes,
                    num_nodes, 
                    num_steps, 
                    type_2opt
                );
            };

            // Execute tasks in parallel
            parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
        }
    }
    
    // Return the optimized tours (already modified in-place)
    return tours;
}

/**
 * @brief PyBind11 module for 2-opt local search
 * 
 * Exposes the two_opt_local_search function to Python.
 */
PYBIND11_MODULE(tsp_2opt_impl, m) {
    m.doc() = "PyBind11 implementation of 2-opt for TSP (GenSCO variant).";
    
    m.def(
        "two_opt_local_search",
        &two_opt_local_search,
        py::arg("tours"),
        py::arg("dists"),
        py::arg("num_steps"),
        py::arg("type_2opt"),
        py::arg("num_workers"),
        "Run 2-opt local search given a distance matrix.\n"
        "type_2opt=2: Find the best pair in each iteration.\n"
        "type_2opt=1: Return as soon as an improving pair is found."
    );
}
