#include <stdio.h>
#include <string.h>
#include <iostream>
#include <argtable3.h>

#include "timer.h"
#include "ils/ils.h"
#include "ils/local_search.h"
#include "mis_log.h"
#include "graph_io.h"
#include "reduction_evolution.h"
#include "mis_config.h"
#include "greedy_mis.h"
#include "parse_parameters.h"
#include "data_structure/graph_access.h"
#include "data_structure/mis_permutation.h"
#include "mis/kernel/ParFastKer/fast_reductions/src/full_reductions.h"
#include "mis_evo.h"

EVO_EXPORT void mis_evo(
    int n, int* xadj, int* adjncy, double time_limit, int* output
){
    graph_access G;
    G.build_from_metis(n, xadj, adjncy);

    MISConfig mis_config;
    configuration_mis cfg;
    cfg.standard(mis_config);
    mis_config.time_limit = time_limit;

    mis_log::instance()->set_graph(G);
    mis_log::instance()->set_config(mis_config);
    
    // Print setup information
    // mis_log::instance()->print_graph();
    // mis_log::instance()->print_config();

    reduction_evolution<branch_and_reduce_algorithm> evo;
    evo.perform_evo_mis(mis_config, G);

    forall_nodes(G, node) {
        if (G.getPartitionIndex(node) == 1) {
            output[node] = 1;
        } 
        else output[node] = 0;
    } endfor
}