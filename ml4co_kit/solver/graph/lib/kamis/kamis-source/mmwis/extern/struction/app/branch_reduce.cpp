/******************************************************************************
 * branch_reduce.cpp
 *
 * Copyright (C) 2015-2017 Sebastian Lamm <lamm@ira.uka.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <argtable3.h>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <random>

#include "timer.h"
#include "mis_log.h"
#include "graph_access.h"
#include "graph_io.h"
#include "mmwis_config.h"
#include "parse_parameters.h"
#include "struction_branch_and_reduce_algorithm.h"


bool is_IS(graph_access& G) {
    forall_nodes(G, node) {
                if (G.getPartitionIndex(node) == 1) {
                    forall_out_edges(G, edge, node) {
                                NodeID neighbor = G.getEdgeTarget(edge);
                                if (G.getPartitionIndex(neighbor) == 1) {
                                    return false;
                                }
                            } endfor
                }
            } endfor

    return true;
}

void assign_weights(graph_access& G, const ::mmwis::MISConfig& mis_config) {
    constexpr NodeWeight MAX_WEIGHT = 200;

    if (mis_config.weight_source == ::mmwis::MISConfig::Weight_Source::HYBRID) {
        forall_nodes(G, node) {
                    G.setNodeWeight(node, (node + 1) % MAX_WEIGHT + 1);
                } endfor
    } else if (mis_config.weight_source == ::mmwis::MISConfig::Weight_Source::UNIFORM) {
        std::default_random_engine generator(mis_config.seed);
        std::uniform_int_distribution<NodeWeight> distribution(1,MAX_WEIGHT);

        forall_nodes(G, node) {
                    G.setNodeWeight(node, distribution(generator));
                } endfor
    } else if (mis_config.weight_source == ::mmwis::MISConfig::Weight_Source::GEOMETRIC) {
        std::default_random_engine generator(mis_config.seed);
        std::binomial_distribution<int> distribution(MAX_WEIGHT / 2);

        forall_nodes(G, node) {
                    G.setNodeWeight(node, distribution(generator));
                } endfor
    }
}

int main(int argn, char **argv) {
    bool output_convergence = false;
    bool output_best_solution = true;

#ifdef OUTPUT_WEIGHT_CONVERGENCE
    std::cout << "is_weight,best_is_time" << std::endl;
        std::cout << "0,0" << std::endl;
#endif

    mis_log::instance()->restart_total_timer();

    ::mmwis::MISConfig mis_config;
    std::string graph_filepath;

    // Parse the command line parameters;
    int ret_code = parse_parameters(argn, argv, mis_config, graph_filepath);
    if (ret_code) {
        return 0;
    }

    mis_config.graph_filename = graph_filepath.substr(graph_filepath.find_last_of('/') + 1);
    mis_log::instance()->set_config(mis_config);

    // Read the graph
    graph_access G;
    graph_io::readGraphWeighted(G, graph_filepath);
    assign_weights(G, mis_config);

    mis_log::instance()->set_graph(G);

    struction::branch_and_reduce_algorithm reducer(G, mis_config);
    auto start = std::chrono::system_clock::now();

#ifndef OUTPUT_WEIGHT_CONVERGENCE
    struction::cout_handler::disable_cout();
#endif
    reducer.run_branch_reduce();
#ifndef OUTPUT_WEIGHT_CONVERGENCE
    struction::cout_handler::enable_cout();
#endif

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> branch_reduce_time = end - start;

    reducer.apply_branch_reduce_solution(G);

    NodeWeight is_weight = 0;
    if (!is_IS(G)) {
        std::cerr << "ERROR: graph after inverse reduction is not independent" << std::endl;
        exit(1);
    } else {
        forall_nodes(G, node) {
                    if (G.getPartitionIndex(node) == 1) {
                        is_weight += G.getNodeWeight(node);
                    }
                } endfor
    }

    std::string name = mis_config.graph_filename.substr(0,mis_config.graph_filename.find_last_of('.'));
#ifndef OUTPUT_WEIGHT_CONVERGENCE
    std::cout << name << "," << G.number_of_nodes() << "," << G.number_of_edges() << ","
              << is_weight << "," << reducer.best_is_time << "," << branch_reduce_time.count() << ","
              << reducer.timeout << ","
              << reducer.min_kernel << "," << reducer.max_min_kernel_comp << "," << reducer.kernelization_time << std::endl;
#endif
#ifdef OUTPUT_WEIGHT_CONVERGENCE
    if (!reducer.timeout) {
            std::cout << "%optimal" << std::endl;
    }
#endif


    if (mis_config.write_solution) {
        std::ofstream f(mis_config.output_filename.c_str());
        std::cout << "writing independent set to " << mis_config.output_filename << " ... " << std::endl;

        forall_nodes(G, node) {
            f << G.getPartitionIndex(node) <<  std::endl;
        } endfor

        f.close();
    } 

    return 0;
}
