#ifndef MCMC_IMPL_H
#define MCMC_IMPL_H

#include <vector>

std::vector<int> mis_mcmc_cpp(
    const int* edge_index,
    int edge_num,
    int nodes_num,
    const double* nodes_weight,
    const int* init_sol,
    const double* taus,
    int steps,
    double penalty_coeff
);

std::vector<int> tsp_mcmc_cpp(
    const double* points,
    int nodes_num,
    int dim,
    const int* init_sol,
    const double* taus,
    int steps
);

std::vector<int> cvrp_mcmc_cpp(
    const double* coords,
    int nodes_num,
    int dim,
    const double* norm_demands,
    const int* init_sol,
    int sol_len,
    const double* taus,
    int steps,
    double penalty_coeff
);

#endif
