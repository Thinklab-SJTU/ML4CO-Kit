#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace {
double euclid_dist(const double* points, int dim, int i, int j) {
    double s = 0.0;
    for (int k = 0; k < dim; ++k) {
        const double d = points[i * dim + k] - points[j * dim + k];
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
}  // namespace

std::pair<std::vector<int>, std::vector<double>> tsp_mcmc_cpp(
    const double* points,
    int nodes_num,
    int dim,
    const int* init_sol,
    const double* taus,
    int steps,
    int seed,
    bool return_trace
) {
    std::vector<int> cur(init_sol, init_sol + nodes_num + 1);
    std::vector<int> best = cur;
    double best_len = tsp_len(points, dim, best);
    double energy = 0.0;
    std::vector<double> trace;
    if (return_trace) {
        trace.reserve(steps);
    }

    std::mt19937 rng(seed);
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

        double tau = taus[t];
        if (tau < 1e-8) {
            tau = 1e-8;
        }
        if (delta >= 0.0 || uni01(rng) < std::exp(delta / tau)) {
            std::reverse(cur.begin() + i, cur.begin() + j + 1);
            energy += delta;
        }

        double cur_len = tsp_len(points, dim, cur);
        if (cur_len < best_len) {
            best_len = cur_len;
            best = cur;
        }
        if (return_trace) {
            trace.push_back(energy);
        }
    }
    return std::make_pair(best, trace);
}
