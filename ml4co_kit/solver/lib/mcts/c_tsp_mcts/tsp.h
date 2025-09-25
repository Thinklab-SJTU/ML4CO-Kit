#ifndef TSP_H
#define TSP_H

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
int max_depth;
double *coord_x;
double *coord_y;
int **distance;
double **edge_heatmap;

// hyper parameters 
double alpha = 1;
double beta = 10;
double param_h = 10;
float time_limit = 1.0;

// city_info
struct Node
{
    int pre_city;
    int next_city;
};
struct Node *all_node;
struct Node *best_all_node;   
int *candidate_num; 
int **candidate;
bool *if_city_selected;

// mcts & 2opt
double begin_time;
int best_distance;
int promising_city_num;
int total_simulation_times;
double avg_weight;
int pair_city_num;
int temp_pair_num;
double **weight;
int **chosen_times;
int *promising_city;
int *probabilistic;
int *city_sequence;
int *temp_city_sequence;
int *gain;
int *real_gain;
int max_iterations_2opt;
int version_2opt;

// solution
int *solution;  


// -------------------- UTILS FUNCTION --------------------- // 
// RANDOM
extern int get_random_num(int range);
// CALCULATE DISTANCE
extern double calculate_double_distance(int city_1,int city_2);
extern int calculate_int_distance(int city_1, int city_2);
extern void calculate_all_pair_distance();
extern int get_distance(int city_1,int city_2);
extern int get_solution_total_distance();
extern double get_current_solution_double_distance();
// CONVERSION
extern void convert_solution_to_all_node();
extern bool convert_all_node_to_solution();
// CHECK
extern bool check_solution_feasible();
extern bool check_if_two_city_same_or_adjacent(int city_1, int city_2);
// SELECT & candidate 
extern int get_best_unselected_city(int cur_city);
extern void identify_candidate_set();
// STORE & RESTORE
extern void store_best_solution();
extern void restore_best_solution();
// REVERSE
extern void reverse_sub_path(int city_1,int city_2);


// --------------------- INIT FUNCTION --------------------- //
// Allocate Memory
extern void allocate_memory(int city_num);
// Release Memory
extern void release_memory(int city_num);
//Estimate the potential of each edge by upper bound confidence function
extern double temp_get_potential(int city_1, int city_2);
// Indentify the promising cities as candidates which are possible to connect to cur_city
extern void temp_identify_promising_city(); 
// Set the probability (stored in probabilistic[]) of selecting each candidate city 
// (proportion to the potential of the corresponding edge)
extern bool temp_get_probabilistic(int cur_city);
// probabilistically choose a city, controled by the values stored in probabilistic[] 
extern int temp_probabilistic_get_city_to_connect();
// The whole process of choosing a city (a_{i+1} in the paper) to connect cur_city (b_i in the paper)
extern int temp_choose_city_to_connect(int cur_city);
// Generate Initial Solution
extern bool generate_initial_solution();


// --------------------- 2OPT FUNCTION --------------------- //
// Evaluate the delta after applying a 2-opt move (delta >0 indicates an improving solution)
extern int get_2opt_delta(int city_1, int city_2);
// Apply a chosen 2-opt move
extern void apply_2opt_move(int city_1,int city_2);
extern bool improve_by_2opt_move();
// Iteratively apply an improving 2-opt move until no improvement is possible
extern void local_search_by_2opt_move();


// ----------------- MARKOV DECISION PROGRESS ---------------- //
extern int mdp();


// -------------------------- MCTS -------------------------- //
// READ DATA
extern void read_heatmap(float *heatmap);
extern void read_nodes_coords(float *nodes_coords);
// INIT
extern void mcts_init();
// GET POTENTIAL
extern double get_avg_weight(int cur_city);
double get_potential(int city_1, int city_2);
void identify_promising_city(int cur_city, int begin_city);
bool get_probabilistic(int cur_city);
extern int probabilistic_get_city_to_connect();
// SIMULATION
// the whole process of choosing a city (a_{i+1} in the paper) to connect cur_city (b_i in the paper)
extern int choose_city_to_connect(int cur_city, int begin_city);
// generate an action starting form begin_city (corresponding to a_1 in the paper), return the delta value
extern int get_simulated_action_delta(int begin_city);
// if the delta of an action is greater than zero, use the information of this action 
//(stored in city_sequence[]) to update the parameters by back propagation
extern void back_propagation(int before_simulation_distance, int action_delta);
//Execute the best action stored in city_sequence[] with depth pair_city_num
extern bool execute_best_action();
// MCTS 
extern void mcts();


#endif  // TSP_H