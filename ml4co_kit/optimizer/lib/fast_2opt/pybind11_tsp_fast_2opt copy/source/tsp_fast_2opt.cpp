#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cassert>

#include "parallel.hpp"
#include "tsp.hpp"

namespace py = pybind11;

inline py::array_t<int> fast_two_opt_local_search(
    py::array_t<int>& tour,
    const py::array_t<float>& points,
    const int num_steps,
    const int near_num_max,
    const int num_workers,
    const uint64_t seed
) {
    assert(tour.flags() & py::array::c_style);
    assert(points.flags() & py::array::c_style);
    assert(near_num_max >= 2);
    assert(tour.ndim() == 1);
    assert(points.ndim() == 2);
    const int num_nodes = static_cast<int>(points.shape()[0]);
    const int point_dim = static_cast<int>(points.shape()[1]);
    assert(num_nodes > 0);
    assert(point_dim >= 2);
    assert(static_cast<int>(tour.shape()[0]) == num_nodes + 1);

    auto* tour_ptr = static_cast<int*>(tour.request().ptr);
    const auto* points_ptr = points.data();
    {
        py::gil_scoped_release release;
        tsp::FastTwoOptEngine engine(points_ptr, num_nodes, point_dim, near_num_max, num_workers);
        engine.run(tour_ptr, num_steps, seed);
    }
    return tour;
}

PYBIND11_MODULE(tsp_fast_2opt_impl, m) {
    m.doc() = "Fast 2-opt local search with GA-EAX style initialization.";
    m.def(
        "fast_two_opt_local_search",
        &fast_two_opt_local_search,
        py::arg("tour"),
        py::arg("points"),
        py::arg("num_steps"),
        py::arg("near_num_max") = 50,
        py::arg("num_workers") = 1,
        py::arg("seed") = static_cast<uint64_t>(1234)
    );
}
