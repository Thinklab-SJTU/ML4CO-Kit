/**
 * three_improve.cpp
 * Purpose: Apply the local search algorithm to a maximum independent set.
 *
 * The original code from Andrade et. al. was kindly provided by Renato Werneck.
 *
 *****************************************************************************/

#include "ils/three_improve.h"

#include <algorithm>

#include "random_functions.h"

three_improve::three_improve() {

}

three_improve::~three_improve() {

}

void three_improve::preprocess_graph(graph_access & G) {
    perm.construct(G); 
    build_candidates(G);
    onetight.clear();
    onetight.resize(G.getMaxDegree());
}

void three_improve::preprocess_graph_with_candidates(graph_access & G, std::vector<NodeID> cand, unsigned int cand_size) {
    perm.construct(G); 
    candidates.init(G.number_of_nodes());
    insert_candidates(G, cand, cand_size);
    onetight.clear();
    onetight.resize(G.getMaxDegree());
}

void three_improve::force(graph_access & G, unsigned int k) {
    for (unsigned int i = 0; i < k; ++i) {
        unsigned int upper_limit = G.number_of_nodes() - perm.get_solution_size();
        unsigned int random = random_functions::nextInt(0, upper_limit - 1);
        NodeID x = perm.get_non_solution_node(random);
        force_node(G, x);
        make_maximal(G);
        direct_improvement(G, true, x);
    }
}

void three_improve::force_node(graph_access & G, NodeID node) {
    forall_out_edges(G, edge, node) {
        NodeID w = G.getEdgeTarget(edge);
        if (perm.is_solution_node(w)) {
            perm.remove_from_solution(w, G);
        }
    } endfor

    perm.add_to_solution(node, G);
    if (!candidates.contains(node)) candidates.insert(node);
}

void three_improve::direct_improvement(graph_access & G, bool forced, NodeID forced_node) {
    
    if (candidates.is_empty()) build_candidates(G);

    while(1) {

        NodeID u;
        if (candidates.is_empty()) {
            if (forced) {
                u = forced_node;
                forced = false;
            }
            else break;
        } else {
            u = candidates.remove_random();
            if (perm.is_solution_node(u)) continue;
            if (forced && u == forced_node) continue;
        }

        // check the current node u
        int u_tight;
        u_tight = check_u(u, G);
        if (u_tight != 2) continue;

        // Build L(x)
        build_onetight(u_x, u_y, u, G);

        if (onetight_size < 2) continue;
        int solutions_found = 0;
        NodeID improv_v = 0;
        NodeID improv_w = 0;

        int v_idx;
        int w_idx;
        for (v_idx = 0; v_idx < onetight_size-1; v_idx++) {
            NodeID v = onetight[v_idx];
            
            // Get the neighbors of v for quick lookup
            std::unordered_set<NodeID> v_neighbors;          
            forall_out_edges(G, e, v) {
                v_neighbors.insert(G.getEdgeTarget(e));
            } endfor

            // find w
            for (int w_idx = v_idx+1; w_idx < onetight_size; w_idx++) {
                NodeID w = onetight[w_idx];
                if (v_neighbors.find(w) == v_neighbors.end()) {
                    solutions_found++;
                    if (random_functions::nextInt(1, solutions_found) == 1) {
                        improv_v = v;
                        improv_w = w;
                    }
                }
            }
        }

        if (solutions_found > 0) {
            perm.remove_from_solution(u_x, G);
            perm.remove_from_solution(u_y, G);
            perm.add_to_solution(improv_v, G);
            perm.add_to_solution(improv_w, G);
            perm.add_to_solution(u, G);
            if (!candidates.contains(u_x)) candidates.insert(u_x);
            if (!candidates.contains(u_y)) candidates.insert(u_y);

            if (!perm.is_maximal()) make_maximal(G);  
            
            // Incremental
            update_candidates(u_x, u_y, G);
        }
        ASSERT_TRUE(perm.check_permutation());
    }
}

void three_improve::insert_candidates(graph_access & G, std::vector<NodeID> cand, unsigned int num_cand) {
    for (unsigned int i = 0; i < num_cand; ++i) {
        if (!candidates.contains(cand[i])) {
            candidates.insert(cand[i]);
        }
    }
}

void three_improve::make_maximal(graph_access & G) {
    while(perm.get_free_size() > 0) {
        int random = random_functions::nextInt(0, perm.get_free_size() - 1);
        NodeID free_node = perm.get_free_node(random);
        perm.add_to_solution(free_node, G);
        if (!candidates.contains(free_node)) candidates.insert(free_node);
    }
}

void three_improve::build_onetight(NodeID node_x, NodeID node_y, NodeID node_u, graph_access & G) {
    onetight_size = 0; 
    std::set<NodeID> onetight_set;

    // Get the neighbors of u for quick lookup
    std::unordered_set<NodeID> u_neighbors;
    forall_out_edges(G, e, node_u) {
        u_neighbors.insert(G.getEdgeTarget(e));
    } endfor

    // node x
    forall_out_edges(G, edge, node_x) {
        NodeID target = G.getEdgeTarget(edge);
        int target_tight = perm.get_tightness(target);
        if (target_tight == 1 && u_neighbors.find(target) == u_neighbors.end()) {
            onetight_set.insert(target);
        }
    } endfor

    // node y
    forall_out_edges(G, edge, node_y) {
        NodeID target = G.getEdgeTarget(edge);
        int target_tight = perm.get_tightness(target);
        if (target_tight == 1 && u_neighbors.find(target) == u_neighbors.end()) {
            onetight_set.insert(target);
        }
    } endfor

    // Copy the set back to the array
    for (NodeID node : onetight_set) {
        onetight[onetight_size++] = node;
    }
}

void three_improve::print_onetight() {
    for (unsigned int i = 0; i < onetight_size; ++i) {
        printf("Node: %d\n", onetight[i]);
    }
}

void three_improve::build_candidates(graph_access & G) {
    candidates.init(G.number_of_nodes());
    NodeID tmp_node;
    for (tmp_node = 0; tmp_node < G.number_of_nodes(); ++tmp_node) {
        if (!perm.is_solution_node(tmp_node)) {
            candidates.insert(tmp_node);
        }
    }
}

void three_improve::update_candidates(NodeID node_x, NodeID node_y, graph_access & G) {
    forall_out_edges(G, edge, node_x) {
        NodeID target = G.getEdgeTarget(edge);
        if (!perm.is_solution_node(target) && perm.get_tightness(target) == 2) {
            if (!candidates.contains(target)) {
                candidates.insert(target);
            }
        }
    } endfor
    
    forall_out_edges(G, edge, node_y) {
        NodeID target = G.getEdgeTarget(edge);
        if (!perm.is_solution_node(target) && perm.get_tightness(target) == 2) {
            if (!candidates.contains(target)) {
                candidates.insert(target);
            }
        }
    } endfor
}

void three_improve::print_permutation() {
    perm.print(0);
    perm.check_permutation();
}

int three_improve::check_u(NodeID node, graph_access & G) {
    int u_tight = 0;
    forall_out_edges(G, edge, node) {
        NodeID target = G.getEdgeTarget(edge);
        if (perm.is_solution_node(target)) {
            if (u_tight == 0) {
                u_x = target;
                u_tight++;
            }
            else if (u_tight == 1) {
                u_y = target;
                u_tight++;
            }
            else {
                return 3;
            }
        }
    } endfor
    return u_tight;
}