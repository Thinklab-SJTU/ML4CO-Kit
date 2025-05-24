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
#include "ils_interface.h"

ILS_EXPORT void arw(int n, int m, int* xadj, int* adjncy, int* initial_solution, int iterations, int* output)
{
    graph_access G;
    G.build_from_metis(n, xadj, adjncy);

    forall_nodes(G, node) {
        G.setPartitionIndex(node, initial_solution[node]);
    } endfor

    ils iterate;
    MISConfig mis_config;
    configuration_mis cfg;
    cfg.standard(mis_config);
    iterate.perform_ils(mis_config, G, iterations);

    forall_nodes(G, node) {
        if (G.getPartitionIndex(node) == 1) {
            output[node] = 1;
        } 
        else output[node] = 0;
    } endfor
}