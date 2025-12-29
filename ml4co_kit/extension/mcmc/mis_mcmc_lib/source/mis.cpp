#include "mis.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <set>

// Propose flip move - randomly select a node to flip
static int proposal_flip(
    int nodes_num,
    std::mt19937& rng,
    std::uniform_int_distribution<int>& int_dist
) {
    return int_dist(rng);
}

// Calculate delta for flip move
static double delta_flip(
    const int* cur_sol,
    int p_node,
    int nodes_num,
    const int* adj_matrix,
    const double* weights,
    double penalty_coeff
) {
    // Calculate conflict weight with neighbors
    double conflict_weight = 0.0;
    for (int i = 0; i < nodes_num; ++i) {
        if (adj_matrix[p_node * nodes_num + i] == 1 && cur_sol[i] == 1) {
            conflict_weight += weights[i];
        }
    }
    conflict_weight *= penalty_coeff;
    
    // Calculate the delta based on whether we're adding or removing the node
    if (cur_sol[p_node] == 1) {
        // Removing node from independent set
        return -weights[p_node] + conflict_weight;
    } else {
        // Adding node to independent set
        return weights[p_node] - conflict_weight;
    }
}

// Apply flip move
static void apply_flip(
    int* cur_sol,
    int p_node
) {
    cur_sol[p_node] = 1 - cur_sol[p_node];
}

// Enhanced MCMC function with flexible return options
std::pair<std::pair<std::vector<std::vector<double>>, bool>, std::vector<double>> mis_mcmc_enhanced(
    const int* adj_matrix,
    const double* weights,
    int nodes_num,
    const int* init_sol,
    double init_cost,
    double penalty_coeff,
    const double* tau_array,
    int tau_length,
    int steps,
    const char* return_type,
    bool return_cost_list
) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::uniform_int_distribution<int> int_dist(0, nodes_num - 1);
    
    // Initialize current solution (copy from init_sol)
    std::vector<int> cur_sol(init_sol, init_sol + nodes_num);
    
    // Initialize cost tracking
    double current_cost = init_cost;
    
    // Best solution tracking
    std::vector<int> best_sol = cur_sol;
    double best_cost = init_cost;
    
    // For mean_sol: accumulate node values
    std::vector<double> node_sum(nodes_num, 0.0);
    
    // For better_sol_list and all_sol_list
    std::vector<std::vector<int>> sol_list;
    
    // For better_sol_list: use set to track unique solutions
    std::set<std::vector<int>> unique_better_sols;
    
    // Cost list tracking
    std::vector<double> cost_list;
    if (return_cost_list) {
        cost_list.reserve(steps);
    }
    
    // Convert return_type to string for comparison
    std::string return_type_str(return_type);
    
    // Determine if we need to track all solutions
    bool track_all = (return_type_str == "all_sol_list");
    bool track_better = (return_type_str == "better_sol_list");
    bool track_mean = (return_type_str == "mean_sol");
    
    // Run MCMC for specified number of steps
    for (int iter = 0; iter < steps; ++iter) {
        // Get temperature for this step
        double tau = (tau_length == 1) ? tau_array[0] : tau_array[iter];
        
        // Propose flip move
        int p_node = proposal_flip(nodes_num, rng, int_dist);
        
        // Calculate delta
        double delta = delta_flip(
            cur_sol.data(), 
            p_node, 
            nodes_num, 
            adj_matrix, 
            weights, 
            penalty_coeff
        );
        
        // Metropolis-Hastings acceptance
        if (delta > 0 || uniform_dist(rng) < std::exp(delta / tau)) {
            apply_flip(cur_sol.data(), p_node);
            current_cost += delta;
        }
        
        // Track cost if requested
        if (return_cost_list) {
            cost_list.push_back(current_cost);
        }
        
        // Update best solution
        if (current_cost > best_cost) {
            best_cost = current_cost;
            best_sol = cur_sol;
        }
        
        // Accumulate for mean solution
        if (track_mean) {
            for (int i = 0; i < nodes_num; ++i) {
                node_sum[i] += cur_sol[i];
            }
        }
        
        // Track solutions for list returns
        if (track_all) {
            sol_list.push_back(cur_sol);
        } else if (track_better && current_cost >= init_cost) {
            // Use set to automatically ensure uniqueness
            unique_better_sols.insert(cur_sol);
        }
    }
    
    // Prepare return values based on return_type
    std::vector<std::vector<double>> result_sols;
    bool is_mean_sol = false;
    
    if (return_type_str == "final_sol") {
        // Convert int solution to double
        std::vector<double> sol_double(cur_sol.begin(), cur_sol.end());
        result_sols.push_back(sol_double);
    } else if (return_type_str == "best_sol") {
        // Convert int solution to double
        std::vector<double> sol_double(best_sol.begin(), best_sol.end());
        result_sols.push_back(sol_double);
    } else if (return_type_str == "mean_sol") {
        // Mean solution as double (probabilities)
        std::vector<double> mean_sol(nodes_num);
        for (int i = 0; i < nodes_num; ++i) {
            mean_sol[i] = node_sum[i] / steps;
        }
        result_sols.push_back(mean_sol);
        is_mean_sol = true;
    } else if (return_type_str == "better_sol_list") {
        // Add initial solution to the unique set
        unique_better_sols.insert(std::vector<int>(init_sol, init_sol + nodes_num));
        
        // Convert set to vector (automatically unique), then to double
        for (const auto& sol : unique_better_sols) {
            std::vector<double> sol_double(sol.begin(), sol.end());
            result_sols.push_back(sol_double);
        }
    } else if (return_type_str == "all_sol_list") {
        // Convert all solutions to double
        for (const auto& sol : sol_list) {
            std::vector<double> sol_double(sol.begin(), sol.end());
            result_sols.push_back(sol_double);
        }
    } else {
        // Default to final_sol
        std::vector<double> sol_double(cur_sol.begin(), cur_sol.end());
        result_sols.push_back(sol_double);
    }
    
    return std::make_pair(std::make_pair(result_sols, is_mean_sol), cost_list);
}

