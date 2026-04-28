#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

namespace py = pybind11;

std::pair<std::vector<int>, std::vector<double>> mis_mcmc_cpp(
    const int* edge_index,
    int edge_num,
    int nodes_num,
    const double* nodes_weight,
    const int* init_sol,
    const double* taus,
    int steps,
    double penalty_coeff,
    int seed,
    bool return_trace
);

py::object mis_mcmc(
    py::array_t<int> edge_index_py,
    py::array_t<double> nodes_weight_py,
    py::array_t<int> init_sol_py,
    py::array_t<double> taus_py,
    double penalty_coeff,
    bool return_trace,
    int seed
) {
    auto edge_buf = edge_index_py.request();
    auto w_buf = nodes_weight_py.request();
    auto init_buf = init_sol_py.request();
    auto tau_buf = taus_py.request();

    int* edge_ptr = static_cast<int*>(edge_buf.ptr);
    double* w_ptr = static_cast<double*>(w_buf.ptr);
    int* init_ptr = static_cast<int*>(init_buf.ptr);
    double* tau_ptr = static_cast<double*>(tau_buf.ptr);

    if (edge_buf.ndim != 2) {
        throw std::runtime_error("edge_index must be a 2D array with shape (2, E) or (E, 2).");
    }
    const bool is_2_by_e = (edge_buf.shape[0] == 2);
    const bool is_e_by_2 = (edge_buf.shape[1] == 2);
    if (!is_2_by_e && !is_e_by_2) {
        throw std::runtime_error("edge_index must have one dimension equal to 2.");
    }

    std::vector<int> edge_index_2xe;
    int edge_num = 0;
    if (is_2_by_e) {
        edge_num = static_cast<int>(edge_buf.shape[1]);
        auto edge = edge_index_py.unchecked<2>();
        edge_index_2xe.assign(2 * edge_num, 0);
        for (int e = 0; e < edge_num; ++e) {
            edge_index_2xe[e] = edge(0, e);
            edge_index_2xe[edge_num + e] = edge(1, e);
        }
        edge_ptr = edge_index_2xe.data();
    } else {
        edge_num = static_cast<int>(edge_buf.shape[0]);
        auto edge = edge_index_py.unchecked<2>();
        edge_index_2xe.assign(2 * edge_num, 0);
        for (int e = 0; e < edge_num; ++e) {
            edge_index_2xe[e] = edge(e, 0);
            edge_index_2xe[edge_num + e] = edge(e, 1);
        }
        edge_ptr = edge_index_2xe.data();
    }
    const int nodes_num = static_cast<int>(w_buf.shape[0]);
    const int steps = static_cast<int>(tau_buf.shape[0]);

    auto result = mis_mcmc_cpp(
        edge_ptr,
        edge_num,
        nodes_num,
        w_ptr,
        init_ptr,
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

PYBIND11_MODULE(mis_mcmc_impl, m) {
    m.doc() = "Simple pybind11 MIS MCMC implementation";
    m.def(
        "mis_mcmc",
        &mis_mcmc,
        py::arg("edge_index"),
        py::arg("nodes_weight"),
        py::arg("init_sol"),
        py::arg("taus"),
        py::arg("penalty_coeff") = 1.001,
        py::arg("return_trace") = false,
        py::arg("seed") = 1234
    );
}
