#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

namespace py = pybind11;

std::pair<std::vector<int>, std::vector<double>> tsp_mcmc_cpp(
    const double* points,
    int nodes_num,
    int dim,
    const int* init_sol,
    const double* taus,
    int steps,
    int seed,
    bool return_trace
);

py::object tsp_mcmc(
    py::array_t<double> points_py,
    py::array_t<int> init_sol_py,
    py::array_t<double> taus_py,
    bool return_trace,
    int seed
) {
    auto points_buf = points_py.request();
    auto init_buf = init_sol_py.request();
    auto tau_buf = taus_py.request();

    double* points_ptr = static_cast<double*>(points_buf.ptr);
    int* init_ptr = static_cast<int*>(init_buf.ptr);
    double* tau_ptr = static_cast<double*>(tau_buf.ptr);

    const int nodes_num = static_cast<int>(points_buf.shape[0]);
    const int dim = static_cast<int>(points_buf.shape[1]);
    const int steps = static_cast<int>(tau_buf.shape[0]);

    auto result = tsp_mcmc_cpp(points_ptr, nodes_num, dim, init_ptr, tau_ptr, steps, seed, return_trace);
    if (return_trace) {
        return py::make_tuple(result.first, result.second);
    }
    return py::cast(result.first);
}

PYBIND11_MODULE(tsp_mcmc_impl, m) {
    m.doc() = "Simple pybind11 TSP MCMC implementation";
    m.def(
        "tsp_mcmc",
        &tsp_mcmc,
        py::arg("points"),
        py::arg("init_sol"),
        py::arg("taus"),
        py::arg("return_trace") = false,
        py::arg("seed") = 1234
    );
}
