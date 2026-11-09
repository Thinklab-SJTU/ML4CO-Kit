#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tsp_greedy.hpp"
#include "parallel.hpp"

namespace py = pybind11;

// Type alias for 32-bit integer
using int32 = int32_t;

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

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
 * @param points Optional node coordinates, (num_nodes, 2) or (batch_size, num_nodes, 2) in batched mode.
 *               Used to join the remaining fragments when the candidate edges do not form a tour.
 * 
 * @return Tour(s):
 *         - Single instance: 1D array of shape (num_nodes + 1,)
 *         - Batched: 2D array of shape (batch_size, num_nodes + 1)
 */
inline py::array_t<int> tsp_greedy_insert(
    const IntArray & candidate_edges,
    const int num_nodes,
    const int num_workers,
    const std::optional<DoubleArray> & points
) {
    const bool batched = candidate_edges.ndim() == 2;
    const int batch_size = batched ? static_cast<int>(candidate_edges.shape()[0]) : 1;
    const long edge_len = static_cast<long>(candidate_edges.shape()[candidate_edges.ndim() - 1]);
    if (candidate_edges.ndim() > 2 || edge_len % 2 != 0) {
        throw std::invalid_argument(
            "candidate_edges must be a flattened edge list [i, j, i, j, ...] of shape (E,) or (B, E) with E even"
        );
    }
    const long num_candidate_edges = edge_len / 2;
    const int* edges_ptr = candidate_edges.data();
    for (long t = 0; t < static_cast<long>(candidate_edges.size()); ++t) {
        if (edges_ptr[t] < 0 || edges_ptr[t] >= num_nodes) {
            throw std::out_of_range(
                "candidate_edges contains node index " + std::to_string(edges_ptr[t]) +
                " outside [0, " + std::to_string(num_nodes) + ")"
            );
        }
    }

    const double* points_ptr = nullptr;
    long points_stride = 0;
    if (points.has_value()) {
        const auto & p = points.value();
        const bool shared = p.ndim() == 2;
        if (!((shared || (batched && p.ndim() == 3 && p.shape()[0] == batch_size)) &&
              p.shape()[p.ndim() - 2] == num_nodes && p.shape()[p.ndim() - 1] == 2)) {
            throw std::invalid_argument("points must have shape (num_nodes, 2), or (batch_size, num_nodes, 2) when batched");
        }
        points_ptr = p.data();
        points_stride = shared ? 0 : 2L * num_nodes;
    }

    // Allocate output tour array(s)
    std::vector<py::ssize_t> shape = {num_nodes + 1};
    if (batched) {
        shape.insert(shape.begin(), batch_size);
    }
    py::array_t<int> tours(shape);
    auto tours_ptr = static_cast<int *>(tours.request().ptr);
    std::vector<char> success(batch_size, 0);

    {
        // Release GIL for better performance during computation
        pybind11::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            success[task_id] = tsp::greedy_insert(
                tours_ptr + static_cast<long>(task_id) * (num_nodes + 1),
                edges_ptr + static_cast<long>(task_id) * num_candidate_edges * 2,
                num_nodes,
                num_candidate_edges,
                points_ptr == nullptr ? nullptr : points_ptr + task_id * points_stride
            );
        };

        if (batched) {
            // Execute tasks in parallel
            parallelize(task_fn, batch_size, std::max(1, std::min(batch_size, num_workers)));
        } else {
            task_fn(0);
        }
    }

    if (std::find(success.begin(), success.end(), 0) != success.end()) {
        throw std::runtime_error(
            "candidate_edges do not connect all nodes into one tour; pass points to join the remaining fragments"
        );
    }
    return tours;
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
        py::arg("points") = py::none(),
        "Construct TSP tour(s) from candidate edges using greedy insertion.\n"
        "\n"
        "Parameters:\n"
        "  candidate_edges: Edge list as 1D array (E,) or 2D array (B, E)\n"
        "                   where E is even and each pair represents an edge (i, j)\n"
        "  num_nodes: Number of nodes in the TSP instance\n"
        "  num_workers: Number of worker threads for parallel execution\n"
        "  points: Optional coordinates (N, 2) or (B, N, 2), used to join the fragments left\n"
        "          when the candidate edges do not form a tour\n"
        "\n"
        "Returns:\n"
        "  Tour(s) as 1D array (N+1,) or 2D array (B, N+1)"
    );
}
