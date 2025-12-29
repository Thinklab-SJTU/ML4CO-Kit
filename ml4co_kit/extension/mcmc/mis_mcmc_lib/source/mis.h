#ifndef MIS_MCMC_H
#define MIS_MCMC_H

#include <vector>
#include <random>
#include <cmath>
#include <utility>

// Enhanced MIS MCMC with flexible return options
// Returns based on return_type: "final_sol", "mean_sol", "best_sol", "better_sol_list", "all_sol_list"
// Note: mean_sol returns double values (probabilities), others return int values (0/1)
std::pair<std::pair<std::vector<std::vector<double>>, bool>, std::vector<double>> mis_mcmc_enhanced(
    const int* adj_matrix,        // Adjacency matrix (nodes_num x nodes_num)
    const double* weights,        // Node weights (nodes_num)
    int nodes_num,                // Number of nodes
    const int* init_sol,          // Initial solution (nodes_num, read-only)
    double init_cost,             // Initial objective value
    double penalty_coeff,         // Penalty coefficient for conflicts
    const double* tau_array,      // Temperature array (steps) or single value
    int tau_length,               // Length of tau_array (1 for constant, steps for variable)
    int steps,                    // Number of MCMC iterations
    const char* return_type,      // Return type: "final_sol", "mean_sol", "best_sol", "better_sol_list", "all_sol_list"
    bool return_cost_list         // Whether to return cost list
);
// Returns: pair<pair<solutions, is_mean_sol>, cost_list>
// - solutions: vector of solutions (as double arrays)
// - is_mean_sol: true if return_type is "mean_sol", false otherwise

#endif // MIS_MCMC_H

