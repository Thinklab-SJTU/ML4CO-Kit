#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

// Type alias for 32-bit integer
using int32 = int32_t;

/**
 * @brief Two-opt local search algorithms for ATSP
 */
namespace atsp {

/**
 * @brief Operation type for ATSP 2-opt moves
 */
enum class OperationType {
    RELOCATE = 0,  // Move node q to after node p
    SWAP = 1       // Swap nodes at positions p and q
};

/**
 * @brief Find the best two-opt move in the current tour for ATSP
 * 
 * Searches through all possible two-opt operations and finds the one that gives
 * the best improvement (or first improvement if type_2opt=1).
 * 
 * For ATSP, we consider two types of operations:
 * 1. RELOCATE: Insert node at position q directly after position p (no path reversal)
 * 2. SWAP: Swap the nodes at positions p and q
 * 
 * @param tour Current tour array (size: num_nodes)
 * @param dist_mat Distance matrix (size: num_nodes * num_nodes, row-major)
 * @param num_nodes Number of nodes in the ATSP instance
 * @param type_2opt Type of 2-opt search:
 *                  - 1: Return as soon as an improving pair is found
 *                  - 2: Find the best pair in the entire search space
 * 
 * @return Tuple of (best_p, best_q, best_delta, best_op_type) where:
 *         - best_p: Index of first position
 *         - best_q: Index of second position
 *         - best_delta: Improvement value (positive means improvement)
 *         - best_op_type: Type of operation (RELOCATE or SWAP)
 */
inline std::tuple<int, int, float, OperationType> find_best_two_opt(
    const int* tour, 
    const float* dist_mat, 
    const int num_nodes, 
    const int type_2opt
) {
    // Initialize with worst possible delta
    float best_delta = std::numeric_limits<float>::lowest();
    int best_p = 0;
    int best_q = 0;
    OperationType best_op_type = OperationType::RELOCATE;

    // Loop through all possible pairs of positions (p, q)
    for (int p = 0; p < num_nodes - 1; ++p) {
        for (int q = p + 1; q < num_nodes; ++q) {
            // Calculate RELOCATE delta: move node at position q to after position p
            // Remove edges: (tour[p] -> tour[p+1]) and (tour[q-1] -> tour[q]) and (tour[q] -> tour[q+1])
            // Add edges: (tour[p] -> tour[q]) and (tour[q] -> tour[p+1]) and (tour[q-1] -> tour[q+1])
            float relocate_delta = 0.0f;
            if (q >= p + 2) {  // Only valid if q is not immediately after p
                relocate_delta =
                    dist_mat[tour[p] * num_nodes + tour[p + 1]] +
                    dist_mat[tour[q - 1] * num_nodes + tour[q]] +
                    dist_mat[tour[q] * num_nodes + tour[q + 1]] -
                    dist_mat[tour[p] * num_nodes + tour[q]] -
                    dist_mat[tour[q] * num_nodes + tour[p + 1]] -
                    dist_mat[tour[q - 1] * num_nodes + tour[q + 1]];
            }
            
            // Calculate SWAP delta: swap nodes at positions p and q
            // Only consider swap if p > 0 (don't swap the starting node)
            // Remove edges: (tour[p-1] -> tour[p]), (tour[p] -> tour[p+1]),
            //               (tour[q-1] -> tour[q]), (tour[q] -> tour[q+1])
            // Add edges: (tour[p-1] -> tour[q]), (tour[q] -> tour[p+1]),
            //            (tour[q-1] -> tour[p]), (tour[p] -> tour[q+1])
            float swap_delta = 0.0f;
            
            // If p == 0, swap_delta remains 0.0f (don't consider swap)
            if (p > 0) {
                // Only calculate swap delta when p > 0
                if (q == p + 1) {
                    // Adjacent nodes: special case
                    // Remove: (tour[p-1] -> tour[p]), (tour[p] -> tour[q]), (tour[q] -> tour[q+1])
                    // Add: (tour[p-1] -> tour[q]), (tour[q] -> tour[p]), (tour[p] -> tour[q+1])
                    swap_delta =
                        dist_mat[tour[p - 1] * num_nodes + tour[p]] +
                        dist_mat[tour[p] * num_nodes + tour[q]] +
                        dist_mat[tour[q] * num_nodes + tour[q + 1]] -
                        dist_mat[tour[p - 1] * num_nodes + tour[q]] -
                        dist_mat[tour[q] * num_nodes + tour[p]] -
                        dist_mat[tour[p] * num_nodes + tour[q + 1]];
                } else {
                    // Non-adjacent nodes
                    swap_delta =
                        dist_mat[tour[p - 1] * num_nodes + tour[p]] +
                        dist_mat[tour[p] * num_nodes + tour[p + 1]] +
                        dist_mat[tour[q - 1] * num_nodes + tour[q]] +
                        dist_mat[tour[q] * num_nodes + tour[q + 1]] -
                        dist_mat[tour[p - 1] * num_nodes + tour[q]] -
                        dist_mat[tour[q] * num_nodes + tour[p + 1]] -
                        dist_mat[tour[q - 1] * num_nodes + tour[p]] -
                        dist_mat[tour[p] * num_nodes + tour[q + 1]];
                }
            }
            
            // Choose the better operation
            float current_delta = std::max(relocate_delta, swap_delta);
            OperationType current_op_type = (relocate_delta > swap_delta) 
                                            ? OperationType::RELOCATE 
                                            : OperationType::SWAP;
            
            if (current_delta > best_delta) {
                // If type_2opt=1, return immediately when finding an improving pair
                if (type_2opt == 1 && current_delta > 0.) {
                    return {p, q, current_delta, current_op_type};
                }
                
                // Update the best pair
                best_delta = current_delta;
                best_p = p;
                best_q = q;
                best_op_type = current_op_type;
            }
        }
    }

    return {best_p, best_q, best_delta, best_op_type};
}

/**
 * @brief Apply a relocate operation to the tour for ATSP
 * 
 * Moves the node at position q to directly after position p.
 * Shifts all elements between p+1 and q-1 one position to the right.
 * 
 * @param tour Tour array to modify (will be modified in-place)
 * @param p First index (insert position)
 * @param q Second index (node to move)
 */
inline void apply_relocate(int* tour, const int p, const int q) {
    // Save the node to be moved
    const int node_to_move = tour[q];
    
    // Shift elements from p+1 to q-1 one position to the right
    for (int i = q; i > p + 1; --i) {
        tour[i] = tour[i - 1];
    }
    
    // Insert the node right after position p
    tour[p + 1] = node_to_move;
}

/**
 * @brief Apply a swap operation to the tour for ATSP
 * 
 * Swaps the nodes at positions p and q.
 * 
 * @param tour Tour array to modify (will be modified in-place)
 * @param p First index
 * @param q Second index
 */
inline void apply_swap(int* tour, const int p, const int q) {
    // Swap the two nodes
    const int temp = tour[p];
    tour[p] = tour[q];
    tour[q] = temp;
}

/**
 * @brief Apply a two-opt operation to the tour for ATSP
 * 
 * Applies either a relocate or swap operation based on the operation type.
 * 
 * @param tour Tour array to modify (will be modified in-place)
 * @param p First index
 * @param q Second index
 * @param op_type Type of operation to apply (RELOCATE or SWAP)
 */
inline void apply_two_opt(int* tour, const int p, const int q, const OperationType op_type) {
    if (op_type == OperationType::RELOCATE) {
        apply_relocate(tour, p, q);
    } else {
        apply_swap(tour, p, q);
    }
}

/**
 * @brief Perform two-opt local search on an ATSP tour
 * 
 * Iteratively finds and applies the best two-opt move until no improvement
 * is found or the maximum number of steps is reached.
 * 
 * Considers both relocate and swap operations, choosing the better one at each step.
 * 
 * @param tour Tour array to optimize (will be modified in-place)
 * @param dist_mat Distance matrix (size: num_nodes * num_nodes, row-major)
 * @param num_nodes Number of nodes in the ATSP instance
 * @param num_steps Maximum number of iterations to perform
 * @param type_2opt Type of 2-opt search (1: first improvement, 2: best improvement)
 */
inline void two_opt(
    int* tour,
    const float* dist_mat,
    const int num_nodes,
    const int num_steps,
    const int type_2opt
) {
    // Early return for invalid inputs
    if (num_steps <= 0) {
        return;
    }

    // Iteratively apply two-opt moves
    for (int step = 0; step < num_steps; ++step) {
        // Find the best two-opt move (considering both relocate and swap)
        auto [p, q, delta, op_type] = find_best_two_opt(tour, dist_mat, num_nodes, type_2opt);
        
        // Stop if no improvement is found (delta < threshold)
        if (delta < 1e-5f) {
            break;
        }
        
        // Apply the best move
        apply_two_opt(tour, p, q, op_type);
    }
}

}  // namespace atsp

