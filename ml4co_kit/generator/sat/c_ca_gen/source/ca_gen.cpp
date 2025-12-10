/*
Modularity-based k-CNF Generator - PyBind11 Wrapper
Based on the original ca.cpp by J. Giráldez-Cru and J. Levy
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

/**
 * @brief Generate CA (Community Attachment) distribution SAT clauses
 * 
 * @param n Number of variables
 * @param m Number of clauses
 * @param k Number of literals per clause (k-CNF)
 * @param q Modularity parameter (0 < q < 1)
 * @param c Number of communities
 * @param seed Random seed
 * @return std::vector<std::vector<int>> List of clauses, each clause is a list of literals
 */
std::vector<std::vector<int>> generate_ca_clauses(
    int n, int m, int k, double q, int c, int seed
) {
    // Validate inputs
    if (c <= 1) {
        throw std::invalid_argument("c must be greater than 1");
    }
    if (q <= 0 || q >= 1) {
        throw std::invalid_argument("q must be in the interval (0,1)");
    }
    if (k < 2) {
        throw std::invalid_argument("k must be greater than 1");
    }
    if (c < k) {
        throw std::invalid_argument("c must be greater or equal than k");
    }
    if (c * k > n) {
        throw std::invalid_argument("c*k must be less or equal than n");
    }
    
    // Compute probability P
    double P = q + 1.0 / (double)c;
    
    // Initialize random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_real(0.0, 1.0);
    std::uniform_int_distribution<int> dist_int(0, std::numeric_limits<int>::max());
    
    // Result: list of clauses
    std::vector<std::vector<int>> clauses;
    clauses.reserve(m);
    
    // Iterate for each clause
    for (int i = 0; i < m; i++) {
        // n2c is the community of each literal
        std::vector<int> n2c(k, 0);
        
        // Compute community assignment for this clause
        double rd = dist_real(rng);
        if (rd <= P) {
            // All variables in the same community
            int rn = dist_int(rng);
            int community = rn % c;
            for (int j = 0; j < k; j++) {
                n2c[j] = community;
            }
        } else {
            // All variables in distinct communities
            for (int j = 0; j < k; j++) {
                bool used = false;
                int rn;
                do {
                    used = false;
                    rn = dist_int(rng);
                    for (int l = 0; l < j && !used; l++) {
                        if (n2c[l] == (rn % c)) {
                            used = true;
                        }
                    }
                } while (used);
                n2c[j] = rn % c;
            }
        }
        
        // Compute the clause
        std::vector<int> clause(k);
        for (int j = 0; j < k; j++) {
            // Random variable in the community
            // avoiding tautologies with previous literals
            int var;
            bool tautology = false;
            int rn;
            do {
                tautology = false;
                rn = dist_int(rng);
                // Calculate variable range for this community
                // Each community has approximately n/c variables
                int var_start = n2c[j] * n / c;
                int var_end = (n2c[j] + 1) * n / c;
                if (var_end <= var_start) {
                    var_end = var_start + 1;
                }
                int var_range = var_end - var_start;
                var = var_start + (rn % var_range) + 1; // +1 because variables start from 1
                
                // Check for tautology with previous literals
                for (int l = 0; l < j && !tautology; l++) {
                    if (std::abs(clause[l]) == var) {
                        tautology = true;
                    }
                }
            } while (tautology);
            
            // Polarity of the variable (randomly assign positive or negative)
            if (rn > (std::numeric_limits<int>::max() / 2)) {
                var = -var;
            }
            
            clause[j] = var;
        }
        
        clauses.push_back(clause);
    }
    
    return clauses;
}

/**
 * @brief PyBind11 module for CA generator
 */
PYBIND11_MODULE(ca_gen_impl, m) {
    m.doc() = "PyBind11 implementation of CA (Community Attachment) SAT instance generator";
    
    m.def(
        "generate_ca_clauses",
        &generate_ca_clauses,
        py::arg("n"),
        py::arg("m"),
        py::arg("k"),
        py::arg("q"),
        py::arg("c"),
        py::arg("seed"),
        "Generate CA distribution SAT clauses.\n"
        "Parameters:\n"
        "  n: Number of variables\n"
        "  m: Number of clauses\n"
        "  k: Number of literals per clause (k-CNF)\n"
        "  q: Modularity parameter (0 < q < 1)\n"
        "  c: Number of communities\n"
        "  seed: Random seed\n"
        "Returns:\n"
        "  List of clauses, each clause is a list of literals"
    );
}

