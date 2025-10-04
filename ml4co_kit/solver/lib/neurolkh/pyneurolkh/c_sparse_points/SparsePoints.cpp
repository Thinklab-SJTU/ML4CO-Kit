#include "SparsePoints.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 


extern "C" {
    int* SparsePoints(
        int _nodes_num, float *_coords, int _sparse_factor, float _scale
    ){
        // Params
        int to, max_cost_idx, cands_num;
        double cost, max_cost;

        // Allocate Memory and Read Problem 
        nodes_num = _nodes_num;
        sparse_factor = _sparse_factor;
        scale = _scale;
        allocateMemory();
        readProblem(_coords);
        
        // Step1: Get Candidates
        for (int blue = 0; blue < nodes_num-1; ++blue){
            for(int to = blue+1; to < nodes_num; ++to){
                cost = calDistance(blue, to);

                // Deal with blue
                cands_num = nodes[blue].candidates_num;
                if (cands_num < sparse_factor){
                    nodes[blue].candidates[cands_num].from = blue;
                    nodes[blue].candidates[cands_num].to = to;
                    nodes[blue].candidates[cands_num].cost = cost;
                    nodes[blue].candidates[cands_num].inverse_idx = -1;
                    if (cost > nodes[blue].max_cost){
                        nodes[blue].max_cost = cost;
                        nodes[blue].max_cost_idx_in_candidates = cands_num;
                    }
                    nodes[blue].candidates_num ++;
                }
                else if (cost < nodes[blue].max_cost){
                    max_cost = cost;
                    max_cost_idx = nodes[blue].max_cost_idx_in_candidates;
                    nodes[blue].candidates[max_cost_idx].from = blue;
                    nodes[blue].candidates[max_cost_idx].to = to;
                    nodes[blue].candidates[max_cost_idx].cost = cost;
                    nodes[blue].candidates[max_cost_idx].inverse_idx = -1;
                    for(int i=0; i<sparse_factor; ++i){
                        if(nodes[blue].candidates[i].cost > max_cost){
                            max_cost = nodes[blue].candidates[i].cost;
                            max_cost_idx = i;
                        }
                    }
                    nodes[blue].max_cost = max_cost; 
                    nodes[blue].max_cost_idx_in_candidates = max_cost_idx;  
                }

                // Deal with To
                cands_num = nodes[to].candidates_num;
                if (cands_num < sparse_factor){
                    nodes[to].candidates[cands_num].from = to;
                    nodes[to].candidates[cands_num].to = blue;
                    nodes[to].candidates[cands_num].cost = cost;
                    nodes[to].candidates[cands_num].inverse_idx = -1;
                    if (cost > nodes[to].max_cost){
                        nodes[to].max_cost = cost;
                        nodes[to].max_cost_idx_in_candidates = cands_num;
                    }
                    nodes[to].candidates_num ++;
                }
                else if (cost < nodes[to].max_cost){
                    max_cost = cost;
                    max_cost_idx = nodes[to].max_cost_idx_in_candidates;
                    nodes[to].candidates[max_cost_idx].from = to;
                    nodes[to].candidates[max_cost_idx].to = blue;
                    nodes[to].candidates[max_cost_idx].cost = cost;
                    nodes[to].candidates[max_cost_idx].inverse_idx = -1;
                    for(int i=0; i<sparse_factor; ++i){
                        if(nodes[to].candidates[i].cost > max_cost){
                            max_cost = nodes[to].candidates[i].cost;
                            max_cost_idx = i;
                        }
                    }
                    nodes[to].max_cost = max_cost; 
                    nodes[to].max_cost_idx_in_candidates = max_cost_idx;  
                }
            }
        }

        // Step2: Sort Candidates from low to high
        for (int i = 0; i < nodes_num; ++i) {
            std::sort(
                nodes[i].candidates,
                nodes[i].candidates + nodes[i].candidates_num,
                [](const Candidate& a, const Candidate& b) {
                    return a.cost < b.cost;
                }
            );
        }

        // Step3: Get Inverse Candidates & Generate Output
        int base_idx;
        for (int blue = 0; blue < nodes_num; ++blue) {
            for (int i= 0; i < sparse_factor; ++i){
                to = nodes[blue].candidates[i].to;
                for (int j = 0; j < sparse_factor; ++j){
                    if (nodes[to].candidates[j].to == blue){
                        nodes[blue].candidates[i].inverse_idx = to * sparse_factor + j;
                        break;
                    }
                }
                base_idx = 3 * (blue * sparse_factor + i);
                outputs[base_idx] = nodes[blue].candidates[i].to;
                outputs[base_idx+1] = (int)(nodes[blue].candidates[i].cost);
                outputs[base_idx+2] = nodes[blue].candidates[i].inverse_idx;
            }
        }

        // Release Memory
        releaseMemory();

        // Return
        return outputs;
    }
}
