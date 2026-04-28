#include <cmath>
#include <utility>
#include <random>
#include <vector>

namespace {
bool is_mis_feasible(const std::vector<int>& sol, const std::vector<std::vector<int>>& adj) {
    const int n = static_cast<int>(sol.size());
    for (int u = 0; u < n; ++u) {
        if (!sol[u]) {
            continue;
        }
        for (int v : adj[u]) {
            if (u < v && sol[v]) {
                return false;
            }
        }
    }
    return true;
}
}  // namespace

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
) {
    std::vector<std::vector<int>> adj(nodes_num);
    for (int e = 0; e < edge_num; ++e) {
        const int u = edge_index[e];
        const int v = edge_index[edge_num + e];
        if (u >= 0 && u < nodes_num && v >= 0 && v < nodes_num && u != v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    std::vector<double> weights(nodes_weight, nodes_weight + nodes_num);
    std::vector<int> cur(init_sol, init_sol + nodes_num);
    std::vector<int> best = cur;
    double energy = 0.0;
    double best_energy = 0.0;
    std::vector<double> trace;
    if (return_trace) {
        trace.reserve(steps);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> node_dist(0, nodes_num - 1);

    for (int t = 0; t < steps; ++t) {
        const int u = node_dist(rng);
        double selected_neighbor_weight = 0.0;
        for (int v : adj[u]) {
            if (cur[v]) {
                selected_neighbor_weight += weights[v];
            }
        }
        selected_neighbor_weight *= penalty_coeff;
        double delta = cur[u] ? (-weights[u] + selected_neighbor_weight)
                              : (weights[u] - selected_neighbor_weight);

        double tau = taus[t];
        if (tau < 1e-8) {
            tau = 1e-8;
        }
        if (delta >= 0.0 || uni01(rng) < std::exp(delta / tau)) {
            cur[u] = 1 - cur[u];
            energy += delta;
            if (is_mis_feasible(cur, adj)) {
                if (energy > best_energy) {
                    best_energy = energy;
                    best = cur;
                }
            }
        }
        if (return_trace) {
            trace.push_back(energy);
        }
    }
    return std::make_pair(best, trace);
}
