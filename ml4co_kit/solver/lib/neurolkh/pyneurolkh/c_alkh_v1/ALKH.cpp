#include "ALKH.h"
#include "Node.h"
#include "Read.h"
#include "Memory.h"
#include "Test.h"
#include "Candidates.h"
#include "PriorityQueue.h"
#include "Core.h"
#include <stdio.h>
#include <stdlib.h>


extern "C" {
    double* ALKH(
        int _nodes_num, float *_coords, float *_penalty,
        int _in_candidates_num, int _out_candidates_num,
        float _scale, float _lr, int _initial_period
    ){
        // Allocate Memory and Read Problem 
        nodes_num = _nodes_num;
        base_max_candidates = _in_candidates_num;
        out_candidates_num = _out_candidates_num;
        LR = _lr;
        scale = _scale;
        InitialPeriod = _initial_period;
        allocateMemory();
        readProblem(_coords, _penalty);

        // ALKH Main
        Ascent();
        
        // Genearte Outputs (EXP & Pi)
        struct Node *curNode = FirstNode;
        do {
            outputs[curNode->id] = (double)(curNode -> exp -> id);
            outputs[curNode->id + nodes_num] = (double)(curNode -> pi);
        } while ((curNode = curNode->suc) != FirstNode);

        // Genearte Outputs (Candidates)
        curNode = FirstNode;
        do {
            for(int i=0; i<out_candidates_num; ++i){
                outputs[curNode->id + (i+2)*nodes_num] = curNode->candidates[i].To->id;
            }
        } while ((curNode = curNode->suc) != FirstNode);

        // Release Memory
        releaseMemory();

        return outputs;
    }
}
