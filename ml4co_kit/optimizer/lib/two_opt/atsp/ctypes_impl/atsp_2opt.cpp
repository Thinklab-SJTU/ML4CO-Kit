#include "atsp.h"
#include "2opt.h"
#include "read.h"
#include "memory.h"
#include "utils.h"
#include <iostream>

extern "C" {
	int* atsp_2opt_local_search(
		short* tour, 
		float *dists, 
		int input_city_num, 
		int input_max_iterations_2opt
	){
		srand(RANDOM_SEED);

		city_num = input_city_num;
		max_iterations_2opt = input_max_iterations_2opt;

		allocate_memory(city_num);
		read_distance(dists);
		read_initial_solution(tour);  		    
		local_search_by_2opt_move();
		convert_all_node_to_solution();
		release_memory(city_num);

		return solution;	
	}
}
