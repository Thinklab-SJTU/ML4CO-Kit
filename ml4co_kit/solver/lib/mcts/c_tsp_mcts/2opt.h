#ifndef OPT_H
#define OPT_H

#include "tsp.h"


// Evaluate the delta after applying a 2-opt move (delta >0 indicates an improving solution)
int get_2opt_delta(int city_1, int city_2)
{
	if(check_if_two_city_same_or_adjacent(city_1, city_2)==true)
		return -INF;
	
	int next_city_1 = all_node[city_1].next_city;
	int next_city_2 = all_node[city_2].next_city;
	
	int delta = get_distance(city_1, next_city_1) + get_distance(city_2, next_city_2)
			- get_distance(city_1, city_2) - get_distance(next_city_1, next_city_2);
	
	// Update the chosen_times[][] and total_simulation_times which are used in MCTS			
	chosen_times[city_1][city_2] ++;
	chosen_times[city_2][city_1] ++;		
	chosen_times[next_city_1][next_city_2] ++;
	chosen_times[next_city_2][next_city_1] ++;	
	total_simulation_times++;		
				
	return delta; 
}

// Apply a chosen 2-opt move
void apply_2opt_move(int city_1,int city_2)
{
	int before_distance = get_solution_total_distance();	
	int delta = get_2opt_delta(city_1, city_2);
	
	int next_city_1=all_node[city_1].next_city;
	int next_city_2=all_node[city_2].next_city;
	
	reverse_sub_path(next_city_1,city_2);	
	all_node[city_1].next_city=city_2;
	all_node[city_2].pre_city=city_1;
	all_node[next_city_1].next_city=next_city_2;	
	all_node[next_city_2].pre_city=next_city_1;
	
	// Update the values of matrix weight[][] by back propagation, which would be used in MCTS
	double increase_rate = beta*(pow(2.718, (double)(delta) / (double)(before_distance)) - 1);
	
	weight[city_1][city_2] += increase_rate;
	weight[city_2][city_1] += increase_rate;		
	weight[next_city_1][next_city_2] += increase_rate;
	weight[next_city_2][next_city_1] += increase_rate;		
}


bool improve_by_2opt_move()
{
	bool if_improved=false;	
	for(int i=0;i<city_num;i++){		
		for(int j=0; j<candidate_num[i]; j++){			
			int candidate_city = candidate[i][j];			
			if(get_2opt_delta(i, candidate_city) > 0){
				apply_2opt_move(i, candidate_city);			
				if_improved=true;
				break;					
			}
		}
	}
	return if_improved;
}

// Iteratively apply an improving 2-opt move until no improvement is possible
void local_search_by_2opt_move()
{
	int iter = 0;
	while(improve_by_2opt_move() == true && iter <= max_iterations_2opt){iter = iter + 1;}	
	int cur_solution_total_distance = get_solution_total_distance();		
	if(cur_solution_total_distance < best_distance)
	{
		// Store the information of the best found solution to Struct_Node *Best_all_node
		best_distance = cur_solution_total_distance;	
		store_best_solution();	  		
	}	
} 

#endif