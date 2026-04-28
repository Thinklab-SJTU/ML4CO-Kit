#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

namespace py = pybind11;

std::pair<std::vector<int>, std::vector<double>> cvrp_mcmc_cpp(
    const double* coords,
    int nodes_num,
    int dim,
    const double* norm_demands,
    const int* init_sol,
    int sol_len,
    const double* taus,
    int steps,
    double penalty_coeff,
    int seed,
    bool return_trace
);

py::object cvrp_mcmc(
    py::array_t<double> coords_py,
    py::array_t<double> norm_demands_py,
    py::array_t<int> init_sol_py,
    py::array_t<double> taus_py,
    double penalty_coeff,
    bool return_trace,
    int seed
) {
    auto coords_buf = coords_py.request();
    auto d_buf = norm_demands_py.request();
    auto init_buf = init_sol_py.request();
    auto tau_buf = taus_py.request();

    double* coords_ptr = static_cast<double*>(coords_buf.ptr);
    double* demand_ptr = static_cast<double*>(d_buf.ptr);
    int* init_ptr = static_cast<int*>(init_buf.ptr);
    double* tau_ptr = static_cast<double*>(tau_buf.ptr);

    const int nodes_num = static_cast<int>(d_buf.shape[0]);
    const int dim = static_cast<int>(coords_buf.shape[1]);
    const int sol_len = static_cast<int>(init_buf.shape[0]);
    const int steps = static_cast<int>(tau_buf.shape[0]);

    auto result = cvrp_mcmc_cpp(
        coords_ptr,
        nodes_num,
        dim,
        demand_ptr,
        init_ptr,
        sol_len,
        tau_ptr,
        steps,
        penalty_coeff,
        seed,
        return_trace
    );
    if (return_trace) {
        return py::make_tuple(result.first, result.second);
    }
    return py::cast(result.first);
}

PYBIND11_MODULE(cvrp_mcmc_impl, m) {
    m.doc() = "Simple pybind11 CVRP MCMC implementation";
    m.def(
        "cvrp_mcmc",
        &cvrp_mcmc,
        py::arg("coords"),
        py::arg("norm_demands"),
        py::arg("init_sol"),
        py::arg("taus"),
        py::arg("penalty_coeff") = 1.001,
        py::arg("return_trace") = false,
        py::arg("seed") = 1234
    );
}
