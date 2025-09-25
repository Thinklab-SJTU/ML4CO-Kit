#ifndef OPT_H
#define OPT_H

#include "atsp.h"


// Evaluate the delta after applying a 2-opt move (delta >0 indicates an improving solution)
int get_2opt_delta(int city_1, int city_2)
{
	if(check_if_two_city_same_or_adjacent(city_1, city_2)==true)
		return -INF;
	
	int next_city_1 = all_node[city_1].next_city;
	int pre_city_2 = all_node[city_2].pre_city;
	int next_city_2 = all_node[city_2].next_city;
	
	int delta = get_distance(city_1, next_city_1) + get_distance(pre_city_2, city_2) + get_distance(city_2, next_city_2)
			- get_distance(city_1, city_2) - get_distance(city_2, next_city_1) - get_distance(pre_city_2, next_city_2);
		
	return delta; 
}

// Apply a chosen 2-opt move
void apply_2opt_move(int city_1, int city_2)
{	
	int next_city_1 = all_node[city_1].next_city;
	int pre_city_2 = all_node[city_2].pre_city;
	int next_city_2 = all_node[city_2].next_city;
	
	all_node[city_1].next_city = city_2;
	all_node[city_2].pre_city = city_1;
	all_node[city_2].next_city = next_city_1;
	all_node[next_city_1].pre_city = city_2;
	all_node[pre_city_2].next_city = next_city_2;
	all_node[next_city_2].pre_city = pre_city_2;	
}


bool improve_by_2opt_move()
{
	bool if_improved = false;
	int best_i = 0;	
	int best_j = 0;
	int best_delta = 0;
	int cur_delta;
	for(int i=0; i<city_num; i++){
		for(int j=0; j<city_num; j++){			
			cur_delta = get_2opt_delta(i, j);
			if(cur_delta > best_delta){
				best_delta = cur_delta;	
				best_j = j;
				best_i = i;				
			}
		}
	}
	if (best_delta > 0){
		if_improved = true;
		apply_2opt_move(best_i, best_j);
	}
	return if_improved;
}

// Iteratively apply an improving 2-opt move until no improvement is possible
void local_search_by_2opt_move()
{
	int iter = 0;
	while(improve_by_2opt_move() == true && iter <= max_iterations_2opt)
	{
		iter = iter + 1;
	}
} 

#endif