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
#include "arw_interface.h"

ARW_EXPORT void arw_1iter(int n, int m, int* xadj, int* adjncy, int* initial_solution, int* output)
{
    graph_access G;
    G.build_from_metis(n, xadj, adjncy);

    forall_nodes(G, node) {
        G.setPartitionIndex(node, initial_solution[node]);
    } endfor

    local_search local;

    local.preprocess_graph(G);
    local.make_maximal(G);
    local.direct_improvement(G);

    forall_nodes(G, node) {
        if (G.getPartitionIndex(node) == 1) {
            output[node] = 1;
        } 
        else output[node] = 0;
    } endfor
}

// int main(int argn, char **argv) {
//     int n, m;
//     std::cin >> n >> m;

//     int * xadj = new int[n + 1];
//     int * adjncy = new int[2 * m];
//     for (int i = 0; i < n + 1; i++) {
//         std::cin >> xadj[i];
//     }
//     for (int i = 0; i < 2 * m; i++) {
//         std::cin >> adjncy[i];
//     }

//     int * independent_set = new int[n];
//     for (int i = 0; i < n; i++) {
//         std::cin >> independent_set[i];
//     }

//     graph_access G;
//     G.build_from_metis(n, xadj, adjncy);

//     forall_nodes(G, node) {
//         G.setPartitionIndex(node, independent_set[node]);
//     } endfor

//     local_search local;

//     local.preprocess_graph(G);
//     local.make_maximal(G);
//     local.direct_improvement(G);

//     forall_nodes(G, node) {
//         if (G.getPartitionIndex(node) == 1) {
//             independent_set[node] = 1;
//         } 
//         else independent_set[node] = 0;
//     } endfor

//     for (int i = 0; i < n; i++) {
//         std::cout << independent_set[i] << ' ';
//     }
//     delete xadj;
//     delete adjncy;
//     delete independent_set;
// }