#ifndef GREEDY_H
#define GREEDY_H

#include "insertion.h"

void greedy_insertion(void)
{	
    int i;
    int j;
    end_idx = 2;
    int node1;
    int node2;
    int cur_node;
    int best_insert_idx;
    double min_cost;
    double cur_cost;
    solution[0] = order[0];
    solution[1] = order[1];
    solution[2] = order[0];

    for(i=2; i<nodes_num; ++i){
        min_cost = 1000000.0;
        cur_node = order[i];

        // find the best_insert_idx
        for(j=0; j<end_idx; ++j){
            node1 = solution[j];
            node2 = solution[j+1];
            cur_cost = get_insert_distance(node1, node2, cur_node);
            if (cur_cost < min_cost){
                best_insert_idx = j;
                min_cost = cur_cost;
            }
        }
        // insert
        for(int k=end_idx+1; k>best_insert_idx+1; --k){
            solution[k] = solution[k-1];
        }
        solution[best_insert_idx+1] = cur_node;
        
        // update end_idx
        end_idx ++;
    }
}


#endif  // GREEDY_H