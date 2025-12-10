/*
Popularity-Similarity SAT Instance Generator - PyBind11 Wrapper
Based on the original ps.cpp by Jesús Giráldez Crú and Jordi Levy
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

// Helper functions
bool myorder(std::pair<int, double> x, std::pair<int, double> y) {
    return x.second < y.second;
}

void getorder(const std::vector<double>& x, std::vector<int>& y, bool inverse) {
    std::vector<std::pair<int, double>> z(x.size());
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        z[i].first = i;
        z[i].second = x[i];
    }
    std::sort(z.begin(), z.end(), myorder);
    if (inverse) {
        for (size_t i = 0; i < x.size(); i++) {
            y[z[i].first] = i;
        }
    } else {
        for (size_t i = 0; i < x.size(); i++) {
            y[i] = z[i].first;
        }
    }
}

template<typename value>
void insertOrdered(std::vector<std::pair<value, double>>& l, std::pair<value, double> elem) {
    size_t pos;
    for (pos = 0; pos < l.size(); pos++) {
        if (elem.second < l[pos].second) {
            break;
        }
    }
    l.resize(l.size() + 1);
    for (size_t i = l.size() - 1; i > pos; i--) {
        l[i] = l[i - 1];
    }
    l[pos] = elem;
}

double myabs(double k) {
    return k >= 0 ? k : -k;
}

/**
 * @brief Generate Popularity-Similarity SAT clauses
 * 
 * @param n Number of variables
 * @param m Number of clauses
 * @param k Average clause size (flexible part, 0 means disabled)
 * @param K Rigid clause size (0 means disabled)
 * @param b Beta for variables
 * @param B Beta for clauses
 * @param T Temperature
 * @param seed Random seed
 * @return std::vector<std::vector<int>> List of clauses, each clause is a list of literals
 */
std::vector<std::vector<int>> generate_ps_clauses(
    int n, int m, int k, int K, double b, double B, double T, int seed
) {
    // Validate inputs
    if (n < 1) {
        throw std::invalid_argument("n (number of nodes) must be greater than 0");
    }
    if (m < 1) {
        throw std::invalid_argument("m (number of clauses) must be greater than 0");
    }
    if (T < 0) {
        throw std::invalid_argument("T (temperature) must be greater or equal than 0");
    }
    if (b < 0 || b > 1 || B < 0 || B > 1) {
        throw std::invalid_argument("b and B (beta) must be in the interval [0,1]");
    }
    
    // Initialize random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_real(0.0, 1.0);
    std::uniform_int_distribution<int> dist_int(0, std::numeric_limits<int>::max());
    
    // Local variables (instead of globals)
    std::vector<std::vector<double>> radius(2);
    std::vector<std::vector<double>> angle(2);
    std::vector<std::vector<int>> neighs(m + 1);
    int edges = 0;
    
    // Resize vectors
    angle[0].resize(n + 1);
    angle[1].resize(m + 1);
    radius[0].resize(n + 1);
    radius[1].resize(m + 1);
    
    // Compute angle location
    for (int i = 1; i <= n; i++) {
        angle[0][i] = dist_real(rng) * 2 * M_PI;
    }
    for (int i = 1; i <= m; i++) {
        angle[1][i] = dist_real(rng) * 2 * M_PI;
    }
    
    // Compute radius location
    for (int i = 1; i <= n; i++) {
        radius[0][i] = dist_real(rng);
    }
    for (int i = 1; i <= m; i++) {
        radius[1][i] = dist_real(rng);
    }
    
    // Helper function to check if edge exists
    auto checkEdge = [&](int i, int j) -> bool {
        return std::find(neighs[j].begin(), neighs[j].end(), i) != neighs[j].end();
    };
    
    // Helper function to create edge
    auto createEdge = [&](int i, int j) {
        if (i <= 0 || i > n || j <= 0 || j > m) {
            throw std::runtime_error("Invalid edge indices");
        }
        if (checkEdge(i, j)) {
            throw std::runtime_error("Edge already exists");
        }
        edges++;
        neighs[j].push_back(i);
    };
    
    // Helper function to compute hyperbolic distance
    auto hyperDist = [&](int i, int j) -> double {
        if (checkEdge(i, j)) return std::numeric_limits<double>::max();
        double diffang = M_PI - myabs(M_PI - myabs(angle[1][j] - angle[0][i]));
        double ri = radius[0][i];
        double rj = radius[1][j];
        return std::pow(ri, b) * std::pow(rj, B) * diffang;
    };
    
    // Add edges for fixed arity (K>0)
    if (K > 0) {
        if (T == 0) {
            for (int j = 1; j <= m; j++) {
                std::vector<std::pair<int, double>> selected;
                for (int i = 1; i <= n; i++) {
                    if (selected.size() < static_cast<size_t>(K)) {
                        insertOrdered(selected, std::make_pair(i, hyperDist(i, j)));
                    } else {
                        double d = hyperDist(i, j);
                        if (d < selected[selected.size() - 1].second) {
                            insertOrdered(selected, std::make_pair(i, d));
                            selected.pop_back();
                        }
                    }
                }
                for (size_t i = 0; i < selected.size(); i++) {
                    createEdge(selected[i].first, j);
                }
            }
        } else { // T > 0
            for (int j = 1; j <= m; j++) {
                int e = 0, eold;
                double SP = 0, SPold;
                
                // Computes a first approximation for sum of probabilities
                for (int i = 1; i <= n; i++) {
                    SP += 1.0 / std::pow(hyperDist(i, j), 1.0 / T);
                }
                
                // Iterates re-calculation of SP until no probability needs being truncated
                do {
                    SPold = SP;
                    eold = e;
                    SP = 0;
                    for (int i = 1; i <= n; i++) {
                        double prob = 1.0 / std::pow(hyperDist(i, j), 1.0 / T);
                        if (prob * (K - eold) / SPold >= 1) {  // If prob needs being truncated, generate corresponding edge
                            if (!checkEdge(i, j)) {
                                createEdge(i, j);
                                e++;
                            }
                        } else {  // else consider probability for next phase
                            SP += prob;
                        }
                    }
                } while (eold < e); // SP=SPOld and e=eold, therefore finish iteration
                
                // Starts random generation of edges
                while (e < K) {
                    int i = dist_int(rng) % n + 1;
                    double prob = 1.0 / std::pow(hyperDist(i, j), 1.0 / T) * (K - eold) / SP;
                    if (!checkEdge(i, j) && dist_real(rng) < prob) {
                        createEdge(i, j);
                        e++;
                    }
                }
            }
        }
    }
    
    // Add edges for flexible arity (k>0)
    if (k > 0) {
        if (T == 0) {
            std::vector<std::pair<std::pair<int, int>, double>> selected;
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= m; j++) {
                    if (!checkEdge(i, j)) {
                        if (selected.size() < static_cast<size_t>(k * m)) {
                            insertOrdered(selected, std::make_pair(std::make_pair(i, j), hyperDist(i, j)));
                        } else {
                            double d = hyperDist(i, j);
                            if (d < selected[selected.size() - 1].second) {
                                insertOrdered(selected, std::make_pair(std::make_pair(i, j), d));
                                selected.pop_back();
                            }
                        }
                    }
                }
            }
            
            for (size_t i = 0; i < selected.size(); i++) {
                createEdge(selected[i].first.first, selected[i].first.second);
            }
        } else { // T > 0
            int e = 0, eold;
            double SP = 0, SPold;
            
            // Computes a first approximation for sum of probabilities
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= m; j++) {
                    SP += 1.0 / std::pow(hyperDist(i, j), 1.0 / T);
                }
            }
            
            // Iterates re-calculation of SP until no probability needs being truncated
            do {
                SPold = SP;
                eold = e;
                SP = 0;
                for (int i = 1; i <= n; i++) {
                    for (int j = 1; j <= m; j++) {
                        if (!checkEdge(i, j)) {
                            double prob = 1.0 / std::pow(hyperDist(i, j), 1.0 / T);
                            if (prob * (k * m - eold) / SPold >= 1) {  // If prob needs being truncated, generate corresponding edge
                                createEdge(i, j);
                                e++;
                            } else {  // else consider probability for next phase
                                SP += prob;
                            }
                        }
                    }
                }
            } while (eold < e); // SP=SPOld and e=eold, therefore finish iteration
            
            // Starts random generation of edges
            while (e < k * m) {
                int i = dist_int(rng) % n + 1;
                int j = dist_int(rng) % m + 1;
                double prob = 1.0 / std::pow(hyperDist(i, j), 1.0 / T) * (k * m - eold) / SP;
                if (!checkEdge(i, j) && dist_real(rng) < prob) {
                    createEdge(i, j);
                    e++;
                }
            }
        }
    }
    
    // Generate clauses from the graph
    std::vector<std::vector<int>> clauses;
    
    std::vector<int> indvar, indcla;
    getorder(radius[0], indvar, true);
    getorder(radius[1], indcla, false);
    
    for (int j = 1; j < static_cast<int>(neighs.size()); j++) {
        int k_idx = indcla[j];
        if (neighs[k_idx].size() > 0) {
            std::vector<int> clause;
            for (size_t i = 0; i < neighs[k_idx].size(); i++) {
                int var = indvar[neighs[k_idx][i]];
                // Randomly assign positive or negative
                if (dist_int(rng) % 2 == 0) {
                    clause.push_back(var);
                } else {
                    clause.push_back(-var);
                }
            }
            clauses.push_back(clause);
        }
    }
    
    return clauses;
}

/**
 * @brief PyBind11 module for PS generator
 */
PYBIND11_MODULE(ps_gen_impl, m) {
    m.doc() = "PyBind11 implementation of Popularity-Similarity SAT instance generator";
    
    m.def(
        "generate_ps_clauses",
        &generate_ps_clauses,
        py::arg("n"),
        py::arg("m"),
        py::arg("k") = 0,
        py::arg("K") = 3,
        py::arg("b") = 0.83,
        py::arg("B") = 1.00,
        py::arg("T") = 1.9,
        py::arg("seed") = 0,
        "Generate Popularity-Similarity SAT clauses.\n"
        "Parameters:\n"
        "  n: Number of variables\n"
        "  m: Number of clauses\n"
        "  k: Average clause size (flexible part, 0 means disabled)\n"
        "  K: Rigid clause size (0 means disabled)\n"
        "  b: Beta for variables\n"
        "  B: Beta for clauses\n"
        "  T: Temperature\n"
        "  seed: Random seed\n"
        "Returns:\n"
        "  List of clauses, each clause is a list of literals"
    );
}

