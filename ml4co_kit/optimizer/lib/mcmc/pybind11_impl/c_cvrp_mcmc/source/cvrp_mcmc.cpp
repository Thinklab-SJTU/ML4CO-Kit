#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace {
double euclid_dist(const double* coords, int dim, int i, int j) {
    double s = 0.0;
    for (int k = 0; k < dim; ++k) {
        const double d = coords[i * dim + k] - coords[j * dim + k];
        s += d * d;
    }
    return std::sqrt(s);
}

double cvrp_penalized_cost(
    const std::vector<int>& sol,
    const double* coords,
    int dim,
    const std::vector<double>& demand,
    double penalty_coeff
) {
    double dist = 0.0;
    for (size_t i = 0; i + 1 < sol.size(); ++i) {
        dist += euclid_dist(coords, dim, sol[i], sol[i + 1]);
    }
    double penalty = 0.0;
    double cur_load = 0.0;
    for (size_t i = 1; i < sol.size(); ++i) {
        const int node = sol[i];
        if (node == 0) {
            if (cur_load > 1.0) {
                penalty += (cur_load - 1.0);
            }
            cur_load = 0.0;
        } else {
            cur_load += demand[node - 1];
        }
    }
    return dist + penalty_coeff * penalty;
}

}  // namespace

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
) {
    std::vector<double> demand(norm_demands, norm_demands + nodes_num);
    std::vector<int> cur(init_sol, init_sol + sol_len);
    std::vector<int> best = cur;
    double best_dist = cvrp_penalized_cost(best, coords, dim, demand, 0.0);
    double cur_cost = cvrp_penalized_cost(cur, coords, dim, demand, penalty_coeff);
    double energy = 0.0;
    std::vector<double> trace;
    if (return_trace) {
        trace.reserve(steps);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    for (int t = 0; t < steps; ++t) {
        std::vector<int> nxt = cur;
        std::vector<int> customer_pos;
        for (int i = 0; i < static_cast<int>(nxt.size()); ++i) {
            if (nxt[i] != 0) {
                customer_pos.push_back(i);
            }
        }
        if (customer_pos.size() < 2) {
            break;
        }

        std::uniform_int_distribution<int> move_dist(0, 2);
        int move = move_dist(rng);

        if (move == 0) {
            std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(customer_pos.size()) - 1);
            int from_idx = customer_pos[pos_dist(rng)];
            int node = nxt[from_idx];
            nxt.erase(nxt.begin() + from_idx);
            std::uniform_int_distribution<int> ins_dist(1, static_cast<int>(nxt.size()) - 1);
            int insert_idx = ins_dist(rng);
            nxt.insert(nxt.begin() + insert_idx, node);
        } else if (move == 1) {
            std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(customer_pos.size()) - 1);
            int p1 = customer_pos[pos_dist(rng)];
            int p2 = customer_pos[pos_dist(rng)];
            while (p1 == p2) {
                p2 = customer_pos[pos_dist(rng)];
            }
            std::swap(nxt[p1], nxt[p2]);
        } else {
            std::uniform_int_distribution<int> p_dist(1, static_cast<int>(nxt.size()) - 2);
            int i = p_dist(rng);
            int j = p_dist(rng);
            if (i > j) {
                std::swap(i, j);
            }
            if (i == j) {
                continue;
            }
            std::reverse(nxt.begin() + i, nxt.begin() + j + 1);
        }

        double nxt_cost = cvrp_penalized_cost(nxt, coords, dim, demand, penalty_coeff);
        double delta = cur_cost - nxt_cost;
        double tau = taus[t];
        if (tau < 1e-8) {
            tau = 1e-8;
        }
        if (delta >= 0.0 || uni01(rng) < std::exp(delta / tau)) {
            cur = std::move(nxt);
            cur_cost -= delta;
            energy += delta;
        }

        double d = cvrp_penalized_cost(cur, coords, dim, demand, 0.0);
        if (d < best_dist) {
            best_dist = d;
            best = cur;
        }
        if (return_trace) {
            trace.push_back(energy);
        }
    }
    return std::make_pair(best, trace);
}
