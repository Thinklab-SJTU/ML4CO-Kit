#ifndef ATSP_H
#define ATSP_H

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define NULL_1             -1 
#define INF                1000000000
#define MAGNITY_RATE       10000
#define MAX_CITY_NUM       10000 
#define MAX_CANDIDATE_NUM  1000 
#define RANDOM_SEED        489663920 


// -------------------- VARIABLE -------------------- // 

// input parameters
int city_num;
int start_city;
int **distance;

// city_info
struct Node
{
    int pre_city;
    int next_city;
};
struct Node *all_node;

// 2opt
int max_iterations_2opt;

// solution
int *solution;


// -------------------- UTILS FUNCTION --------------------- // 
// RANDOM
extern int get_random_num(int range);

// GET DISTANCE
extern int get_distance(int city_1,int city_2);

// CONVERSION
extern void convert_solution_to_all_node();
extern bool convert_all_node_to_solution();

// CHECK
extern bool check_if_two_city_same_or_adjacent(int city_1, int city_2);


// --------------------- MEMORY FUNCTION --------------------- //
// Allocate Memory
extern void allocate_memory(int city_num);

// Release Memory
extern void release_memory(int city_num);


// --------------------- 2OPT FUNCTION --------------------- //
// Evaluate the delta after applying a 2-opt move (delta >0 indicates an improving solution)
extern int get_2opt_delta(int city_1, int city_2);

// Apply a chosen 2-opt move
extern void apply_2opt_move(int city_1,int city_2);
extern bool improve_by_2opt_move();

// Iteratively apply an improving 2-opt move until no improvement is possible
extern void local_search_by_2opt_move();


// -------------------------- READ -------------------------- //
extern void read_heatmap(float *heatmap);
extern void read_initial_solution(short* tour);

#endif 