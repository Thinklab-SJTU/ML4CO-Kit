#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mis.h"

namespace py = pybind11;


// Python wrapper for enhanced MIS MCMC
py::object mis_mcmc_enhanced_impl(
    py::array_t<int> adj_matrix_py,
    py::array_t<double> weights_py,
    py::array_t<int> init_sol_py,
    double penalty_coeff,
    py::object tau_py,  // Can be float or array
    int steps,
    std::string return_type,
    bool return_cost_list
) {
    // Get adjacency matrix
    auto adj_matrix_buf = adj_matrix_py.request();
    int* adj_matrix_ptr = static_cast<int*>(adj_matrix_buf.ptr);
    int nodes_num = static_cast<int>(adj_matrix_buf.shape[0]);
    
    // Get node weights
    auto weights_buf = weights_py.request();
    double* weights_ptr = static_cast<double*>(weights_buf.ptr);
    
    // Get initial solution
    auto init_sol_buf = init_sol_py.request();
    int* init_sol_ptr = static_cast<int*>(init_sol_buf.ptr);
    
    // Calculate initial cost automatically
    double init_cost = 0.0;
    for (int i = 0; i < nodes_num; ++i) {
        if (init_sol_ptr[i] == 1) {
            init_cost += weights_ptr[i];
        }
    }

    // Handle tau (can be scalar or array)
    std::vector<double> tau_vec;
    int tau_length;
    
    if (py::isinstance<py::array>(tau_py)) {
        // tau is an array
        py::array_t<double> tau_array = tau_py.cast<py::array_t<double>>();
        auto tau_buf = tau_array.request();
        tau_length = static_cast<int>(tau_buf.shape[0]);
        
        if (tau_length != steps && tau_length != 1) {
            throw std::runtime_error("Temperature array length must be 1 or equal to steps");
        }
        
        double* tau_ptr = static_cast<double*>(tau_buf.ptr);
        tau_vec.assign(tau_ptr, tau_ptr + tau_length);
    } else {
        // tau is a scalar
        double tau_scalar = tau_py.cast<double>();
        tau_vec.push_back(tau_scalar);
        tau_length = 1;
    }
    
    // Call C++ function
    auto result = mis_mcmc_enhanced(
        adj_matrix_ptr,
        weights_ptr,
        nodes_num,
        init_sol_ptr,
        init_cost,
        penalty_coeff,
        tau_vec.data(),
        tau_length,
        steps,
        return_type.c_str(),
        return_cost_list
    );
    
    // Extract solutions, is_mean_sol flag, and cost list
    std::vector<std::vector<double>>& solutions = result.first.first;
    bool is_mean_sol = result.first.second;
    std::vector<double>& cost_list = result.second;
    
    // Convert solutions to numpy arrays
    py::list sol_list_py;
    for (const auto& sol : solutions) {
        if (is_mean_sol || return_type == "mean_sol") {
            // Return as double array for mean_sol
            py::array_t<double> sol_py(nodes_num);
            auto sol_buf = sol_py.request();
            double* sol_ptr = static_cast<double*>(sol_buf.ptr);
            
            for (int i = 0; i < nodes_num; ++i) {
                sol_ptr[i] = sol[i];
            }
            sol_list_py.append(sol_py);
        } else {
            // Return as int array for other types
            py::array_t<int> sol_py(nodes_num);
            auto sol_buf = sol_py.request();
            int* sol_ptr = static_cast<int*>(sol_buf.ptr);
            
            for (int i = 0; i < nodes_num; ++i) {
                sol_ptr[i] = static_cast<int>(sol[i]);
            }
            sol_list_py.append(sol_py);
        }
    }
    
    // Return based on whether cost_list is requested
    if (return_cost_list) {
        // Convert cost_list to numpy array
        py::array_t<double> cost_list_py(cost_list.size());
        auto cost_buf = cost_list_py.request();
        double* cost_ptr = static_cast<double*>(cost_buf.ptr);
        
        for (size_t i = 0; i < cost_list.size(); ++i) {
            cost_ptr[i] = cost_list[i];
        }
        
        // Return tuple of (solutions, cost_list)
        if (return_type == "final_sol" || return_type == "best_sol" || return_type == "mean_sol") {
            // Return single solution and cost_list
            return py::make_tuple(sol_list_py[0], cost_list_py);
        } else {
            // Return list of solutions and cost_list
            return py::make_tuple(sol_list_py, cost_list_py);
        }
    } else {
        // Return only solutions
        if (return_type == "final_sol" || return_type == "best_sol" || return_type == "mean_sol") {
            // Return single solution
            return sol_list_py[0];
        } else {
            // Return list of solutions
            return sol_list_py;
        }
    }
}

PYBIND11_MODULE(mis_mcmc_lib, m) {
    m.doc() = "Enhanced MIS MCMC with flexible temperature and return options";
    
    m.def("mis_mcmc_enhanced_impl",
          &mis_mcmc_enhanced_impl,
          py::arg("adj_matrix"),
          py::arg("weights"),
          py::arg("init_sol"),
          py::arg("penalty_coeff"),
          py::arg("tau"),
          py::arg("steps"),
          py::arg("return_type") = "final_sol",
          py::arg("return_cost_list") = false,
          "Run enhanced MCMC on MIS with flexible options\n\n"
          "Args:\n"
          "    adj_matrix: Adjacency matrix (nodes_num, nodes_num)\n"
          "    weights: Node weights (nodes_num,)\n"
          "    init_sol: Initial solution (nodes_num,) with 0/1 values\n"
          "    penalty_coeff: Penalty coefficient for conflicts\n"
          "    tau: Temperature (scalar or array of length steps)\n"
          "    steps: Number of MCMC iterations\n"
          "    return_type: Type of solution to return:\n"
          "        'final_sol' - final solution after steps\n"
          "        'best_sol' - best solution encountered\n"
          "        'mean_sol' - average solution\n"
          "        'better_sol_list' - all solutions >= init_cost (including init)\n"
          "        'all_sol_list' - all solutions encountered\n"
          "    return_cost_list: If True, also return cost at each step\n\n"
          "Returns:\n"
          "    If return_cost_list=False:\n"
          "        - Single solution array (for final_sol, best_sol, mean_sol)\n"
          "        - List of solution arrays (for better_sol_list, all_sol_list)\n"
          "    If return_cost_list=True:\n"
          "        - Tuple of (solution(s), cost_list)\n\n"
          "Note:\n"
          "    Initial cost is calculated automatically from init_sol and weights");
}

