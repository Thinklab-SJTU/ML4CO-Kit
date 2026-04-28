#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mcmc.h"

namespace py = pybind11;

std::vector<int> mis_mcmc(
    py::array_t<int> edge_index_py,
    py::array_t<double> nodes_weight_py,
    py::array_t<int> init_sol_py,
    py::array_t<double> taus_py,
    double penalty_coeff
) {
    auto edge_buf = edge_index_py.request();
    auto w_buf = nodes_weight_py.request();
    auto init_buf = init_sol_py.request();
    auto tau_buf = taus_py.request();

    int* edge_ptr = static_cast<int*>(edge_buf.ptr);
    double* w_ptr = static_cast<double*>(w_buf.ptr);
    int* init_ptr = static_cast<int*>(init_buf.ptr);
    double* tau_ptr = static_cast<double*>(tau_buf.ptr);

    const int edge_num = static_cast<int>(edge_buf.shape[1]);
    const int nodes_num = static_cast<int>(w_buf.shape[0]);
    const int steps = static_cast<int>(tau_buf.shape[0]);

    return mis_mcmc_cpp(
        edge_ptr,
        edge_num,
        nodes_num,
        w_ptr,
        init_ptr,
        tau_ptr,
        steps,
        penalty_coeff
    );
}

std::vector<int> tsp_mcmc(
    py::array_t<double> points_py,
    py::array_t<int> init_sol_py,
    py::array_t<double> taus_py
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

    return tsp_mcmc_cpp(points_ptr, nodes_num, dim, init_ptr, tau_ptr, steps);
}

std::vector<int> cvrp_mcmc(
    py::array_t<double> coords_py,
    py::array_t<double> norm_demands_py,
    py::array_t<int> init_sol_py,
    py::array_t<double> taus_py,
    double penalty_coeff
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

    return cvrp_mcmc_cpp(
        coords_ptr,
        nodes_num,
        dim,
        demand_ptr,
        init_ptr,
        sol_len,
        tau_ptr,
        steps,
        penalty_coeff
    );
}

PYBIND11_MODULE(mcmc_impl, m) {
    m.doc() = "Simple pybind11 MCMC implementations for MIS/TSP/CVRP";

    m.def(
        "mis_mcmc",
        &mis_mcmc,
        py::arg("edge_index"),
        py::arg("nodes_weight"),
        py::arg("init_sol"),
        py::arg("taus"),
        py::arg("penalty_coeff") = 1.001
    );
    m.def(
        "tsp_mcmc",
        &tsp_mcmc,
        py::arg("points"),
        py::arg("init_sol"),
        py::arg("taus")
    );
    m.def(
        "cvrp_mcmc",
        &cvrp_mcmc,
        py::arg("coords"),
        py::arg("norm_demands"),
        py::arg("init_sol"),
        py::arg("taus"),
        py::arg("penalty_coeff") = 1.001
    );
}
