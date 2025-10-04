#ifndef READ_PROBLEM_H
#define READ_PROBLEM_H

#include "ALKH.h"

void readProblem(float *_coords, float *_penalty)
{
    // Init Priority Queue
    pq_init(nodes_num);

    // Create Node Links
    FirstNode = createNode(
        0, 
        _coords[0] * scale, 
        _coords[1] * scale, 
        _penalty[0] * scale
    );
    struct Node *LastNode = FirstNode;
    for (int i=1; i<nodes_num; i++){
        struct Node *newNode = createNode(
            i, 
            _coords[2*i] * scale, 
            _coords[2*i+1] * scale,
            _penalty[i] * scale
        );
        LastNode -> suc = newNode;
        newNode -> pred = LastNode;
        if(i == nodes_num -1){
            newNode -> suc = FirstNode;
            FirstNode -> pred = newNode;
        }
        LastNode = newNode;
    }
}

#endif