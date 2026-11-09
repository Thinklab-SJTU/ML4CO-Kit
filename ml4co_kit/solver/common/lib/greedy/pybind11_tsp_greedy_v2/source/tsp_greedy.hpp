#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// Type alias for 32-bit integer
using int32 = int32_t;

/**
 * @brief Greedy insertion algorithm for TSP
 */
namespace tsp {

/**
 * @brief Find the representative of node x (union-find with path halving)
 */
inline int find_root(std::vector<int> & parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/**
 * @brief Insert edges in the given order, skipping those that violate the degree or subtour constraint
 * 
 * @param edges Flattened edge list [i, j, i, j, ...]
 * @param num_edges Number of edges in the list
 * @param num_nodes Number of nodes in the TSP instance
 * @param neighbors Neighbor slots of every node (2 per node, -1 = empty), updated in place
 * @param parent Union-find parents of the path fragments, updated in place
 * @param num_inserted_edges Number of edges inserted so far
 * @return Number of edges inserted after processing the list
 */
inline int insert_edges(
    const int* edges,
    const long num_edges,
    const int num_nodes,
    std::vector<int> & neighbors,
    std::vector<int> & parent,
    int num_inserted_edges
) {
    for (long edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        if (num_inserted_edges == num_nodes - 1) {
            break;
        }
        const int i = edges[2 * edge_idx];
        const int j = edges[2 * edge_idx + 1];

        // Skip self-loops and nodes that already have 2 neighbors (degree constraint)
        if (i == j || neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }

        // Skip edges inside one fragment (premature cycle)
        const int root_i = find_root(parent, i);
        const int root_j = find_root(parent, j);
        if (root_i == root_j) {
            continue;
        }
        parent[root_i] = root_j;
        neighbors[2 * i + (neighbors[2 * i] != -1)] = j;
        neighbors[2 * j + (neighbors[2 * j] != -1)] = i;
        ++num_inserted_edges;
    }
    return num_inserted_edges;
}

/**
 * @brief Join the remaining path fragments by nearest-endpoint greedy insertion
 * 
 * Endpoints (nodes with fewer than 2 neighbors) propose edges to their k nearest endpoints, which are
 * inserted in increasing order of length; k doubles whenever a round inserts nothing. Each round costs
 * O(m^2) for m endpoints, which is small when the candidate list covers most of the tour.
 * 
 * @param points Node coordinates as flattened array [x, y, x, y, ...]
 * @param num_nodes Number of nodes in the TSP instance
 * @param neighbors Neighbor slots of every node (2 per node, -1 = empty), updated in place
 * @param parent Union-find parents of the path fragments, updated in place
 * @param num_inserted_edges Number of edges inserted so far
 * @return Number of inserted edges, always num_nodes - 1 (one Hamiltonian path)
 */
inline int join_fragments(
    const double* points,
    const int num_nodes,
    std::vector<int> & neighbors,
    std::vector<int> & parent,
    int num_inserted_edges
) {
    auto dist = [points](const int a, const int b) {
        const double dx = points[2 * a] - points[2 * b];
        const double dy = points[2 * a + 1] - points[2 * b + 1];
        return std::sqrt(dx * dx + dy * dy);
    };

    int k = 8;
    std::vector<int> ends;
    std::vector<std::pair<double, int>> row;
    std::vector<std::pair<double, long>> order;
    std::vector<int> proposals;
    std::vector<int> sorted_edges;
    while (num_inserted_edges < num_nodes - 1) {
        ends.clear();
        for (int v = 0; v < num_nodes; ++v) {
            if (neighbors[2 * v + 1] == -1) {
                ends.push_back(v);
            }
        }
        const int m = static_cast<int>(ends.size());
        const int kk = std::min(k, m - 1);

        proposals.clear();
        order.clear();
        row.resize(m);
        for (int a = 0; a < m; ++a) {
            for (int b = 0; b < m; ++b) {
                row[b] = {b == a ? std::numeric_limits<double>::infinity() : dist(ends[a], ends[b]), b};
            }
            std::partial_sort(row.begin(), row.begin() + kk, row.end());
            for (int t = 0; t < kk; ++t) {
                order.push_back({row[t].first, static_cast<long>(proposals.size() / 2)});
                proposals.push_back(ends[a]);
                proposals.push_back(ends[row[t].second]);
            }
        }
        std::stable_sort(order.begin(), order.end(), [](const auto & x, const auto & y) {
            return x.first < y.first;
        });
        sorted_edges.resize(proposals.size());
        for (size_t t = 0; t < order.size(); ++t) {
            sorted_edges[2 * t] = proposals[2 * order[t].second];
            sorted_edges[2 * t + 1] = proposals[2 * order[t].second + 1];
        }

        const int before = num_inserted_edges;
        num_inserted_edges = insert_edges(
            sorted_edges.data(), static_cast<long>(order.size()), num_nodes, neighbors, parent, num_inserted_edges
        );
        if (num_inserted_edges == before) {
            k *= 2;
        }
    }
    return num_inserted_edges;
}

/**
 * @brief Greedy insertion algorithm to construct a TSP tour from candidate edges
 * 
 * This algorithm constructs a TSP tour by greedily inserting edges from a sorted
 * list of candidate edges. An edge is inserted if it doesn't violate:
 * 1. Degree constraint: Each node can have at most 2 neighbors
 * 2. Subtour constraint: No premature cycles (until all nodes are connected)
 * If the candidate edges do not connect all nodes into one path (e.g. kNN-sparse heatmaps), the
 * remaining fragments are joined by nearest-endpoint greedy insertion when ``points`` is given.
 * 
 * @param tour Output tour array (size: num_nodes + 1, last element equals first)
 * @param candidate_edges Input edge list as flattened array [i, j, i, j, ...]
 *                        where edge index k represents nodes (candidate_edges[2*k], candidate_edges[2*k+1])
 * @param num_nodes Number of nodes in the TSP instance
 * @param num_candidate_edges Number of candidate edges in the list
 * @param points Node coordinates as flattened array [x, y, x, y, ...], or nullptr
 * @return false if the candidate edges do not form a tour and no points are given (tour is left unset)
 */
inline bool greedy_insert(
    int* tour, 
    const int* candidate_edges, 
    const int num_nodes, 
    const long num_candidate_edges,
    const double* points = nullptr
) {
    // Neighbors of each node (-1 = undetermined), at most 2 per node
    std::vector<int> neighbors(2 * num_nodes, -1);

    // Union-find over the path fragments
    std::vector<int> parent(num_nodes);
    for (int v = 0; v < num_nodes; ++v) {
        parent[v] = v;
    }

    int num_inserted_edges = insert_edges(
        candidate_edges, num_candidate_edges, num_nodes, neighbors, parent, 0
    );
    if (num_inserted_edges < num_nodes - 1) {
        if (points == nullptr) {
            return false;
        }
        num_inserted_edges = join_fragments(points, num_nodes, neighbors, parent, num_inserted_edges);
    }

    // Convert neighbor representation to tour
    // Find a node with only one neighbor (endpoint of the path)
    int start_node = 0;
    while (neighbors[2 * start_node + 1] != -1) {
        ++start_node;
    }
    
    tour[0] = start_node;
    
    // Ensure the start node's single neighbor is in the first position
    const int start_node_neighbor = neighbors[2 * start_node];
    if (num_nodes > 1 && neighbors[2 * start_node_neighbor] == start_node) {
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
            tour[i] = neighbors[2 * current_node + 1];
            current_node = neighbors[2 * current_node + 1];
        }
    }
    
    // Close the tour by returning to the start node
    tour[num_nodes] = tour[0];
    return true;
}

}  // namespace tsp
