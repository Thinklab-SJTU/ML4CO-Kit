#ifndef INSERTION_H
#define INSERTION_H

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// input
int nodes_num;
double *coord_x;
double *coord_y;
int *order;
extern void read_nodes_coords(float *nodes_coords);
extern void read_order(short* input_order);

// memory
extern void allocate_memory(int nodes_num);
extern void release_memory(int nodes_num);

// distance
extern double get_distance(int node1_idx, int node2_idx);
extern double get_insert_distance(int node1_idx, int node2_idx, int insert_node_idx);

// solution & nodes
int *solution;

// greedy insertion
int end_idx;
extern void greedy_insertion(void);

// test
extern void print_num(int x);
extern void print_solution(void);
extern void print_solution_length(void);

#endif  // INSERTION_H