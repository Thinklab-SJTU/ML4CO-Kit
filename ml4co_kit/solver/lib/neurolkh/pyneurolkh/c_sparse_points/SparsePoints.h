#ifndef SparsePoints_H
#define SparsePoints_H

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>


// Params
int nodes_num;
int sparse_factor;
float scale;
int *outputs;

struct Candidate {
    int from;
    int to;
    double cost;
    int inverse_idx;
};

struct Node {
    double x;
    double y;
    double max_cost;
    int max_cost_idx_in_candidates;
    int candidates_num;
    struct Candidate *candidates;
};
struct Node *nodes;


// Memory Related
void allocateMemory(){
    outputs = new int[3*nodes_num*sparse_factor];
    nodes = (struct Node*)malloc(nodes_num * sizeof(struct Node));
}

void releaseMemory(){
    free(nodes);
}


// Read Problem
void readProblem(float *coords)
{
    for(int i = 0; i<nodes_num; ++i){
        nodes[i].x = coords[2*i] * scale;
        nodes[i].y = coords[2*i+1] * scale;
        nodes[i].max_cost = 0;
        nodes[i].candidates_num = 0;
        nodes[i].candidates = (struct Candidate*)malloc(sparse_factor * sizeof(struct Candidate));
    }
}


// Get Distance of nodes
double calDistance(int a_idx, int b_idx)
{
    double dx = nodes[a_idx].x - nodes[b_idx].x;
    double dy = nodes[a_idx].y - nodes[b_idx].y;
    return sqrt(dx * dx + dy * dy);
}


#endif