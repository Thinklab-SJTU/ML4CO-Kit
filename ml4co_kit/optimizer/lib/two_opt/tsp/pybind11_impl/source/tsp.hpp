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
 * @brief Two-opt local search algorithms for TSP
 */
namespace tsp {

/**
 * @brief Find the best two-opt move in the current tour
 * 
 * Searches through all possible two-opt swaps and finds the one that gives
 * the best improvement (or first improvement if type_2opt=1).
 * 
 * @param tour Current tour array (size: num_nodes)
 * @param dist_mat Distance matrix (size: num_nodes * num_nodes, row-major)
 * @param num_nodes Number of nodes in the TSP instance
 * @param type_2opt Type of 2-opt search:
 *                  - 1: Return as soon as an improving pair is found
 *                  - 2: Find the best pair in the entire search space
 * 
 * @return Tuple of (best_i, best_j, delta) where:
 *         - best_i: First index of the best swap
 *         - best_j: Second index of the best swap
 *         - delta: Improvement value (positive means improvement)
 */
inline std::tuple<int, int, float> find_best_two_opt(
    const int* tour, 
    const float* dist_mat, 
    const int num_nodes, 
    const int type_2opt
) {
    // Initialize with worst possible delta
    float best_delta = std::numeric_limits<float>::lowest();
    int best_i = 0;
    int best_j = 0;

    // Loop through all possible pairs of cities (i, j) where j > i+1
    // to avoid invalid swaps (adjacent cities or same city)
    // Note: tour size is num_nodes + 1 (tour[num_nodes] == tour[0], duplicate start node)
    for (int i = 0; i < num_nodes - 1; ++i) {
        for (int j = i + 2; j < num_nodes; ++j) {
            // Calculate the delta (improvement) of swapping edges:
            // Remove: (tour[i] -> tour[i+1]) and (tour[j] -> tour[j+1])
            // Add:    (tour[i] -> tour[j]) and (tour[i+1] -> tour[j+1])
            // Note: tour[j+1] can be directly accessed when j < num_nodes - 1,
            //       and when j == num_nodes - 1, tour[j+1] == tour[num_nodes] == tour[0]
            const float delta =
                dist_mat[tour[i] * num_nodes + tour[i + 1]] +
                dist_mat[tour[j] * num_nodes + tour[j + 1]] -
                dist_mat[tour[i] * num_nodes + tour[j]] -
                dist_mat[tour[i + 1] * num_nodes + tour[j + 1]];
            
            if (delta > best_delta) {
                // If type_2opt=1, return immediately when finding an improving pair
                if (type_2opt == 1 && delta > 0.) {
                    return {i, j, delta};
                }
                
                // Update the best pair
                best_delta = delta;
                best_i = i;
                best_j = j;
            }
        }
    }

    return {best_i, best_j, best_delta};
}

/**
 * @brief Apply a two-opt swap to the tour
 * 
 * Reverses the segment of the tour between indices i+1 and j (inclusive).
 * This effectively swaps the edges (i, i+1) and (j, j+1) with (i, j) and (i+1, j+1).
 * 
 * @param tour Tour array to modify (will be modified in-place)
 * @param i First index of the swap
 * @param j Second index of the swap
 */
inline void apply_two_opt(int* tour, const int i, const int j) {
    // Reverse the segment between i+1 and j (inclusive)
    std::reverse(tour + i + 1, tour + j + 1);
}

/**
 * @brief Perform two-opt local search on a TSP tour
 * 
 * Iteratively finds and applies the best two-opt move until no improvement
 * is found or the maximum number of steps is reached.
 * 
 * @param tour Tour array to optimize (will be modified in-place)
 * @param dist_mat Distance matrix (size: num_nodes * num_nodes, row-major)
 * @param num_nodes Number of nodes in the TSP instance
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
        // Find the best two-opt move
        auto [i, j, delta] = find_best_two_opt(tour, dist_mat, num_nodes, type_2opt);
        
        // Stop if no improvement is found (delta < threshold)
        if (delta < 1e-5f) {
            break;
        }
        
        // Apply the best move
        apply_two_opt(tour, i, j);
    }
}

}  // namespace tsp
