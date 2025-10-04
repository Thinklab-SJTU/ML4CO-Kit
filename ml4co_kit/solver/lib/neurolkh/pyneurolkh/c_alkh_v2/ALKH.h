#ifndef ALKH_H
#define ALKH_H

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

// #define BASE_MAX_CANDIDATES 10

// -------------------- Variables -------------------- // 

int nodes_num;
int Norm;
int pq_size;
double *outputs;
int base_max_candidates;
int out_candidates_num;
int InitialPeriod;
float LR;
float scale;


struct Candidate {
    struct Node *To; // The End Node of the Edge
    double cost; // Cost of the Edge
    double alpha; // Alpha of the Edge (Min means important)
};


struct Node {
    double x;
    double y;
    int id;
    int index; // index in the PriorityQueue (-1 means not in the PQ)
    int max_candidates; // Node's Max Candidates Num
    double pi; // Node Penalty
    double best_pi; // Best Node Penalty
    int degree; // Node Degree Minus 2
    int last_degree; // Last Node Degree Minus 2
    Node *pred; // Predecessor Node
    Node *suc;  // Successor Node
    Node *exp;  // Expected Node (For the Expected Node, this Node is the closest to it.)
    Node *subopt; // Sub-Opt Node (The edge between them is not in the MST)
    double exp_cost; // The Expected Cost (from expected node to itself)
    double real_cost; // The Real Cost (from pred node to itself)
    double subopt_cost; // The Sub-Opt Cost (from itself to subopt node)
    struct Candidate *candidates; // Candidates Array
};


struct Node *FirstNode; // First Node
struct Node **pq; // Priority Queue


// -------------------- Main Functions -------------------- // 
// Memory Related
extern void allocateMemory();
extern void releaseMemory();

// Node Related
extern struct Node* createNode(int id,  float x, float y, float penalty);
extern double getCost(struct Node *a, struct Node *b);
extern void Follow(struct Node *a, struct Node *b);
extern void Precede(struct Node *a, struct Node *b);
extern void freeNodeList(struct Node* FirstNode);
extern void freeNode(struct Node* node);
extern void initNodeCandidates(struct Node* node);
extern void freeNodeCandidates(struct Node* node);

// Candidates Related
extern int compareCandidates(const void *a, const void *b);
extern void createCandidates(int* _candidates);
extern void freeCandidates(void);
extern void sortCandidates(void);

// Read Related
extern void readProblem(float *_coords, float *_penalty);

// Core Related
extern double sparseM1Tree(void);
extern double Ascent(void);

// Priority Queue Related
extern void swap(struct Node **a, struct Node **b);
extern void pq_insert(struct Node *node);
extern void pq_lazy_insert(struct Node *node);
extern struct Node *pq_extract_min();
extern void pq_heapify_up(int i);
extern void pq_heapify_down(int i);
extern void pq_init(int max_size);
extern void pq_free(void);

// Test Related
extern void print_number(int num);
extern void print_nodes_id(void);
extern void print_nodes_info(void);
extern void print_nodes_degrees(void);
extern void print_candidates(void);

#endif