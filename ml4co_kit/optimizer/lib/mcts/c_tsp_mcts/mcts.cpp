#include "tsp.h"
#include "2opt.h"
#include "init.h"
#include "utils.h"
#include "mcts.h"
#include "test.h"
#include <iostream>

extern "C" {
	int* mcts_local_search(
		short* tour, 
		float *heatmap, 
		float *nodes_coords, 
		int input_city_num, 
		int input_max_depth, 
		float input_time_limit,
		int input_version_2opt,
		int input_continue_flag,
		int input_max_iterations_2opt
	){
		srand(RANDOM_SEED);
		city_num = input_city_num;
		max_depth = input_max_depth;
		time_limit = input_time_limit;
		version_2opt = input_version_2opt;
		continue_flag = input_continue_flag;
		max_iterations_2opt = input_max_iterations_2opt;
		begin_time = (double)clock();
		best_distance = INF;   
		allocate_memory(city_num);
		read_heatmap(heatmap);
		read_nodes_coords(nodes_coords);
		read_initial_solution(tour);
		calculate_all_pair_distance();	
		identify_candidate_set(); 	  		    
		mdp();
		convert_all_node_to_solution();
		release_memory(city_num);
		return solution;	
	}
}
