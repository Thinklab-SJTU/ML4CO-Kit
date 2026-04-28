#include "mcmc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <utility>

namespace {
double euclid_dist(const double* coords, int dim, int i, int j) {
    double s = 0.0;
    for (int k = 0; k < dim; ++k) {
        const double d = coords[i * dim + k] - coords[j * dim + k];
        s += d * d;
    }
    return std::sqrt(s);
}

double tsp_len(const double* points, int dim, const std::vector<int>& tour) {
    double total = 0.0;
    for (size_t i = 0; i + 1 < tour.size(); ++i) {
        total += euclid_dist(points, dim, tour[i], tour[i + 1]);
    }
    return total;
}

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

std::vector<int> repair_mis(
    const std::vector<int>& input_sol,
    const std::vector<double>& weights,
    const std::vector<std::vector<int>>& adj
) {
    std::vector<int> sol = input_sol;
    const int n = static_cast<int>(sol.size());
    for (int u = 0; u < n; ++u) {
        if (!sol[u]) {
            continue;
        }
        for (int v : adj[u]) {
            if (u < v && sol[v]) {
                if (weights[u] >= weights[v]) {
                    sol[v] = 0;
                } else {
                    sol[u] = 0;
                    break;
                }
            }
        }
    }
    return sol;
}

double mis_weight(const std::vector<int>& sol, const std::vector<double>& w) {
    double score = 0.0;
    for (size_t i = 0; i < sol.size(); ++i) {
        if (sol[i]) {
            score += w[i];
        }
    }
    return score;
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

bool cvrp_feasible(const std::vector<int>& sol, const std::vector<double>& demand, int nodes_num) {
    std::vector<int> seen(nodes_num + 1, 0);
    if (sol.empty() || sol.front() != 0 || sol.back() != 0) {
        return false;
    }
    double cur_load = 0.0;
    for (size_t i = 1; i < sol.size(); ++i) {
        int node = sol[i];
        if (node == 0) {
            if (cur_load > 1.0 + 1e-8) {
                return false;
            }
            cur_load = 0.0;
            continue;
        }
        if (node < 1 || node > nodes_num) {
            return false;
        }
        seen[node] += 1;
        cur_load += demand[node - 1];
    }
    for (int i = 1; i <= nodes_num; ++i) {
        if (seen[i] != 1) {
            return false;
        }
    }
    return true;
}

std::vector<int> cvrp_repair(const std::vector<int>& sol, const std::vector<double>& demand, int nodes_num) {
    std::vector<int> customers;
    customers.reserve(nodes_num);
    std::vector<int> used(nodes_num + 1, 0);
    for (int x : sol) {
        if (x > 0 && x <= nodes_num && !used[x]) {
            customers.push_back(x);
            used[x] = 1;
        }
    }
    for (int i = 1; i <= nodes_num; ++i) {
        if (!used[i]) {
            customers.push_back(i);
        }
    }

    std::vector<int> repaired;
    repaired.push_back(0);
    double cur_load = 0.0;
    for (int c : customers) {
        const double d = demand[c - 1];
        if (cur_load + d > 1.0 + 1e-8 && repaired.back() != 0) {
            repaired.push_back(0);
            cur_load = 0.0;
        }
        repaired.push_back(c);
        cur_load += d;
    }
    if (repaired.back() != 0) {
        repaired.push_back(0);
    }
    return repaired;
}

double tau_at(const double* taus, int steps, int t) {
    if (steps <= 0) {
        return 1e-6;
    }
    double tau = taus[t];
    if (tau < 1e-8) {
        tau = 1e-8;
    }
    return tau;
}
}  // namespace

std::vector<int> mis_mcmc_cpp(
    const int* edge_index,
    int edge_num,
    int nodes_num,
    const double* nodes_weight,
    const int* init_sol,
    const double* taus,
    int steps,
    double penalty_coeff
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
    std::vector<int> best = repair_mis(cur, weights, adj);
    double best_w = mis_weight(best, weights);

    double energy = 0.0;
    std::random_device rd;
    std::mt19937 rng(rd());
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
        double delta = 0.0;
        if (cur[u]) {
            delta = -weights[u] + selected_neighbor_weight;
        } else {
            delta = weights[u] - selected_neighbor_weight;
        }

        const double tau = tau_at(taus, steps, t);
        if (delta >= 0.0 || uni01(rng) < std::exp(delta / tau)) {
            cur[u] = 1 - cur[u];
            energy += delta;
            (void)energy;
        }

        if (is_mis_feasible(cur, adj)) {
            const double w = mis_weight(cur, weights);
            if (w > best_w) {
                best_w = w;
                best = cur;
            }
        }
    }
    return best;
}

std::vector<int> tsp_mcmc_cpp(
    const double* points,
    int nodes_num,
    int dim,
    const int* init_sol,
    const double* taus,
    int steps
) {
    std::vector<int> cur(init_sol, init_sol + nodes_num + 1);
    std::vector<int> best = cur;
    double best_len = tsp_len(points, dim, best);
    double energy = 0.0;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> i_dist(1, nodes_num - 2);

    for (int t = 0; t < steps; ++t) {
        int i = i_dist(rng);
        std::uniform_int_distribution<int> j_dist(i + 1, nodes_num - 1);
        int j = j_dist(rng);

        int a = cur[i - 1];
        int b = cur[i];
        int c = cur[j];
        int d = cur[j + 1];

        double old_cost = euclid_dist(points, dim, a, b) + euclid_dist(points, dim, c, d);
        double new_cost = euclid_dist(points, dim, a, c) + euclid_dist(points, dim, b, d);
        double delta = old_cost - new_cost;

        double tau = tau_at(taus, steps, t);
        if (delta >= 0.0 || uni01(rng) < std::exp(delta / tau)) {
            std::reverse(cur.begin() + i, cur.begin() + j + 1);
            energy += delta;
            (void)energy;
        }

        double cur_len = tsp_len(points, dim, cur);
        if (cur_len < best_len) {
            best_len = cur_len;
            best = cur;
        }
    }
    return best;
}

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
) {
    std::vector<double> demand(norm_demands, norm_demands + nodes_num);
    std::vector<int> cur(init_sol, init_sol + sol_len);
    std::vector<int> best = cvrp_feasible(cur, demand, nodes_num) ? cur : cvrp_repair(cur, demand, nodes_num);
    double best_dist = cvrp_penalized_cost(best, coords, dim, demand, 0.0);
    double cur_cost = cvrp_penalized_cost(cur, coords, dim, demand, penalty_coeff);
    double energy = 0.0;

    std::random_device rd;
    std::mt19937 rng(rd());
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
        double tau = tau_at(taus, steps, t);
        if (delta >= 0.0 || uni01(rng) < std::exp(delta / tau)) {
            cur = std::move(nxt);
            cur_cost -= delta;
            energy += delta;
            (void)energy;
        }

        if (cvrp_feasible(cur, demand, nodes_num)) {
            double d = cvrp_penalized_cost(cur, coords, dim, demand, 0.0);
            if (d < best_dist) {
                best_dist = d;
                best = cur;
            }
        }
    }

    if (!cvrp_feasible(best, demand, nodes_num)) {
        best = cvrp_repair(best, demand, nodes_num);
    }
    return best;
}
