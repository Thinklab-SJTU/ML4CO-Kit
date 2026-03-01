#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

// Type alias for 32-bit integer
using int32 = int32_t;

/**
 * @brief Greedy insertion algorithm for TSP
 */
namespace tsp {

/**
 * @brief Greedy insertion algorithm to construct a TSP tour from candidate edges
 * 
 * This algorithm constructs a TSP tour by greedily inserting edges from a sorted
 * list of candidate edges. An edge is inserted if it doesn't violate:
 * 1. Degree constraint: Each node can have at most 2 neighbors
 * 2. Subtour constraint: No premature cycles (until all nodes are connected)
 * 
 * @param tour Output tour array (size: num_nodes + 1, last element equals first)
 * @param candidate_edges Input edge list as flattened array [i, j, i, j, ...]
 *                        where edge index k represents nodes (candidate_edges[2*k], candidate_edges[2*k+1])
 * @param num_nodes Number of nodes in the TSP instance
 * @param num_candidate_edges Number of candidate edges in the list
 */
inline void greedy_insert(
    int* tour, 
    const int* candidate_edges, 
    const int num_nodes, 
    const int num_candidate_edges
) {
    // Track which subtour each node belongs to (0 = not inserted yet)
    std::vector<int> subtour_id(num_nodes, 0);
    
    // Lambda to update subtour IDs when merging two subtours
    auto update_id = [&subtour_id, num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subtour_id[i] == from) {
                subtour_id[i] = to;
            }
        }
    };

    // Track neighbors for each node (-1 = undetermined)
    // Each node can have at most 2 neighbors
    std::vector<int> neighbors(2 * num_nodes, -1);
    
    // Lambda to set neighbor relationship between nodes i and j
    auto set_neighbor = [&neighbors](const int i, const int j) {
        // Set j as neighbor of i
        if (neighbors[2 * i] == -1) {
            neighbors[2 * i] = j;
        } else {
            assert(neighbors[2 * i + 1] == -1);
            neighbors[2 * i + 1] = j;
        }
        // Set i as neighbor of j
        if (neighbors[2 * j] == -1) {
            neighbors[2 * j] = i;
        } else {
            assert(neighbors[2 * j + 1] == -1);
            neighbors[2 * j + 1] = i;
        }
    };

    int next_available_subtour = 1;
    int num_inserted_edges = 0;
    
    // Process each candidate edge in order
    for (int edge_idx = 0; edge_idx < num_candidate_edges; ++edge_idx) {
        const int i = candidate_edges[2 * edge_idx];
        const int j = candidate_edges[2 * edge_idx + 1];

        // Skip if either node already has 2 neighbors (degree constraint)
        if (neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }
        
        // Skip self-loops
        if (i == j) {
            continue;
        }

        // Case 1: Node i has not been inserted yet
        if (subtour_id[i] == 0) {
            ++num_inserted_edges;
            if (subtour_id[j] == 0) {
                // Both nodes are new - create a new subtour
                subtour_id[i] = next_available_subtour;
                subtour_id[j] = next_available_subtour;
                ++next_available_subtour;
            } else {
                // Node j is already in a subtour - add i to it
                subtour_id[i] = subtour_id[j];
            }
            set_neighbor(i, j);
        } 
        // Case 2: Node i has been inserted
        else {
            if (subtour_id[j] == 0) {
                // Node j is new - add it to i's subtour
                ++num_inserted_edges;
                subtour_id[j] = subtour_id[i];
                set_neighbor(i, j);
            } else {
                // Both nodes have been inserted
                if (subtour_id[i] == subtour_id[j]) {
                    // Same subtour - would create a premature cycle, skip
                    continue;
                } else {
                    // Different subtours - merge them
                    ++num_inserted_edges;
                    update_id(subtour_id[j], subtour_id[i]);
                    set_neighbor(i, j);
                }
            }
        }

        // Stop when we have a complete tour (n-1 edges for n nodes)
        if (num_inserted_edges == num_nodes - 1) {
            break;
        }
    }

    // Ensure we have a valid tour
    assert(num_inserted_edges == num_nodes - 1);

    // Convert neighbor representation to tour
    // Find a node with only one neighbor (endpoint of the path)
    int start_node = 0;
    while (neighbors[2 * start_node + 1] != -1) {
        ++start_node;
    }
    
    tour[0] = start_node;
    
    // Ensure the start node's single neighbor is in the first position
    const int start_node_neighbor = neighbors[2 * start_node];
    if (neighbors[2 * start_node_neighbor] == start_node) {
        std::swap(neighbors[2 * start_node_neighbor], neighbors[2 * start_node_neighbor + 1]);
    }
    
    int current_node = start_node;
    std::vector<bool> has_visited(num_nodes, false);

    // Build the tour by following neighbors
    for (int i = 1; i < num_nodes; ++i) {
        has_visited[current_node] = true;
        if (!has_visited[neighbors[2 * current_node]]) {
            tour[i] = neighbors[2 * current_node];
            current_node = neighbors[2 * current_node];
        } else {
            assert(!has_visited[neighbors[2 * current_node + 1]]);
            tour[i] = neighbors[2 * current_node + 1];
            current_node = neighbors[2 * current_node + 1];
        }
    }
    
    // Close the tour by returning to the start node
    tour[num_nodes] = tour[0];
}

}  // namespace tsp

