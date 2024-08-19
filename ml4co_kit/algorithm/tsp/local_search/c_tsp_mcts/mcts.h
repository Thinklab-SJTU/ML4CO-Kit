#ifndef MCTS_H
#define MCTS_H

#include "tsp.h"

// ----------------------------- READ DATA ----------------------------- //

void read_heatmap(float *heatmap)
{
    int i;
    for(i=0; i<city_num; i++){  
    	for(int j=0; j<city_num; j++){	
	    	edge_heatmap[i][j] = heatmap[i*city_num + j];
        }
    }	    		
		
	for(i=0; i<city_num; i++){
    	for(int j=i+1; j<city_num; j++)
	    {	    
	    	edge_heatmap[i][j] = (edge_heatmap[i][j] + edge_heatmap[j][i]) / 2;
	    	edge_heatmap[j][i] = edge_heatmap[i][j];	    	    	
		}
    }
}


void read_nodes_coords(float *nodes_coords)
{	
	int i;
	start_city = 0;
	for(i=0; i<city_num; i++)
	{
		coord_x[i]= nodes_coords[2*i] * MAGNITY_RATE;
		coord_y[i]= nodes_coords[2*i+1] * MAGNITY_RATE;			
	}
}


bool read_initial_solution(short *tour)
{
	int i;
	for (i=0; i<city_num; ++i)
		solution[i] = tour[i];
	convert_solution_to_all_node();
	store_best_solution();
	if(check_solution_feasible()==false)
		return false;
	
	return true;		
}


// ----------------------------- INIT  ----------------------------- //

void mcts_init()
{
	for(int i=0; i<city_num; i++){
		for(int j=0; j<city_num; j++){
			weight[i][j] = edge_heatmap[i][j] * 100;
			chosen_times[i][j] = 0;
		}
    }
	total_simulation_times = 0;	
}


// ------------------------------  MDP  ----------------------------- //

int mdp()
{
	mcts_init();                      // Initialize MCTS parameters	
	local_search_by_2opt_move();	  // 2-opt based local search within small neighborhood		
	mcts();		                      // Tageted sampling via MCTS within enlarged neighborhood	
	version_2opt = 2;
	max_iterations_2opt = 5000;
	local_search_by_2opt_move();      // Again 2-opt based local search within small neighborhood		
	restore_best_solution();
	if(check_solution_feasible())
		return get_solution_total_distance();
	else
		return INF;
}


// -------------------------- GET POTENTIAL  ------------------------ //

double get_avg_weight(int cur_city)
{
	double total_weight = 0;
	for(int i=0; i<city_num; i++)	
	{
		if(i==cur_city)
			continue;
		total_weight += weight[cur_city][i];						
	}
		
	return total_weight / (city_num-1);
}

double get_potential(int city_1, int city_2)
{	
    double right_part = sqrt( log( total_simulation_times + 1) / ( log(2.718)*(chosen_times[city_1][city_2]+1) ) );
	double potential = weight[city_1][city_2] / avg_weight + alpha * right_part; 
	return potential;	
}

void identify_promising_city(int cur_city, int begin_city)
{
	promising_city_num = 0;
	for(int i=0; i < candidate_num[cur_city]; i++)	
	{
		int temp_city = candidate[cur_city][i];				
		if(temp_city == begin_city)			
			continue;					
		if(temp_city == all_node[cur_city].next_city)		
			continue;
		if(get_potential(cur_city, temp_city) < 1)	
			continue;
		
		promising_city[promising_city_num++] = temp_city;						
	}
}

bool get_probabilistic(int cur_city)
{
	if(promising_city_num == 0)
		return false;
		
	double total_potential = 0;
	for(int i=0;i<promising_city_num;i++)	
		total_potential += get_potential(cur_city, promising_city[i]);	
	probabilistic[0] = (int)(1000 * get_potential(cur_city, promising_city[0]) / total_potential);
	for(int i=1; i<promising_city_num-1; i++)
		probabilistic[i] = probabilistic[i-1] + (int)(1000 * get_potential(cur_city, promising_city[i]) / total_potential);	
	probabilistic[promising_city_num-1] = 1000;	
	
	return true;
}

// probabilistically choose a city, controled by the values stored in probabilistic[] 
int probabilistic_get_city_to_connect()
{
	int random_num = get_random_num(1000);
	for(int i=0; i<promising_city_num; i++)
		if(random_num < probabilistic[i])
			return promising_city[i];
	return NULL_1;
}


// -------------------------- SIMULATION  ------------------------ //

// the whole process of choosing a city (a_{i+1} in the paper) to connect cur_city (b_i in the paper)
int choose_city_to_connect(int cur_city, int begin_city)
{
	avg_weight = get_avg_weight(cur_city);		
	identify_promising_city(cur_city, begin_city);
	get_probabilistic(cur_city);	
	return probabilistic_get_city_to_connect();
}

// generate an action starting form begin_city (corresponding to a_1 in the paper), return the delta value
int get_simulated_action_delta(int begin_city)
{
	// store the current solution to solution[]
	if(convert_all_node_to_solution() == false)
		return -INF;		
	
	int next_city = all_node[begin_city].next_city;   // a_1=begin city, b_1=next_city
	
	// break edge (a_1,b_1)
	all_node[begin_city].next_city = NULL_1;           
	all_node[next_city].pre_city = NULL_1;				

	// the elements of an action is stored in city_sequence[], where a_{i+1}=city_sequence[2*i], b_{i+1}=city_sequence[2*i+1]
	city_sequence[0] = begin_city;                     
	city_sequence[1] = next_city;	
		
	gain[0] = get_distance(begin_city, next_city);                // gain[i] stores the delta (before connecting to a_1) at the (i+1)th iteration  
	real_gain[0] = gain[0]-get_distance(next_city, begin_city);   // real_gain[i] stores the delta (after connecting to a_1) at the (i+1)th iteration
	pair_city_num = 1;                                            // pair_city_num indicates the depth (k in the paper) of the action	
	
	bool if_changed = false;				
	int cur_city = next_city;	    // b_i = cur_city (1 <= i <= k)
	while(true)
	{		
		int next_city_to_connect = choose_city_to_connect(cur_city, begin_city);	// 	probabilistically choose one city as a_{i+1}
		if(next_city_to_connect == NULL_1)
			break;	
		
		//Update the chosen times, used in mcts	
		chosen_times[cur_city][next_city_to_connect] ++;
		chosen_times[next_city_to_connect][cur_city] ++;	
		
		int next_city_to_disconnect = all_node[next_city_to_connect].pre_city;   // determine b_{i+1}
		
		// Update city_sequence[], gain[], real_gain[] and pair_city_num
		city_sequence[2*pair_city_num] = next_city_to_connect;
		city_sequence[2*pair_city_num+1] = next_city_to_disconnect;				
		gain[pair_city_num] = gain[pair_city_num-1] - get_distance(cur_city, next_city_to_connect) \
                + get_distance(next_city_to_connect, next_city_to_disconnect);
		real_gain[pair_city_num] = gain[pair_city_num]-get_distance(next_city_to_disconnect, begin_city);
		pair_city_num++;					
		
		// reverse the cities between b_i and b_{i+1}	
		reverse_sub_path(cur_city, next_city_to_disconnect);			
		all_node[cur_city].next_city = next_city_to_connect;
		all_node[next_city_to_connect].pre_city = cur_city;
		all_node[next_city_to_disconnect].pre_city = NULL_1;		
		if_changed = true;
		
		// turns to the next iteration
		cur_city=next_city_to_disconnect;
		
		// close the loop is meeting an improving action, or the depth reaches its upper bound	
		if(real_gain[pair_city_num-1] > 0 || pair_city_num > max_depth)		
			break;					
	}
	
	// restore the solution before simulation	
	if(if_changed)	
		convert_solution_to_all_node();			
	else
	{
		all_node[begin_city].next_city = next_city;
		all_node[next_city].pre_city = begin_city;	
	}
	
	// identify the best depth of the simulated action
	int max_real_gain = -INF;
	int best_index = 1;
	for(int i=1; i<pair_city_num; i++)	
		if(real_gain[i] > max_real_gain)
		{
			max_real_gain = real_gain[i];	
			best_index=i;
		}
				
	pair_city_num = best_index+1;
	
	return max_real_gain;	
}

// if the delta of an action is greater than zero, use the information of this action (stored in city_sequence[]) to update the parameters by back propagation
void back_propagation(int before_simulation_distance, int action_delta)
{
	for(int i=0; i<pair_city_num; i++)  
	{
		int city_2 = city_sequence[2*i+1];
		int third_city;
		if(i<pair_city_num-1)	
			third_city = city_sequence[2*i+2];
		else
			third_city = city_sequence[0];
		
		if(action_delta > 0)
		{
			double increase_rate = beta*(pow(2.718, (double) (action_delta) / (double)(before_simulation_distance) )-1);
			weight[city_2][third_city] += increase_rate;
			weight[third_city][city_2] += increase_rate;							
		}		
	}		
}

// sampling at most max_simulation_times actions
int simulation(int max_simulation_times)
{
	int best_action_delta = -INF;		
	for(int i=0; i<max_simulation_times; i++)
	{
		int begin_city = get_random_num(city_num);				
		int action_delta = get_simulated_action_delta(begin_city);		
		total_simulation_times++;
		
		//store the action with the best delta, stored in temp_city_sequence[] and temp_pair_num					
		if(action_delta > best_action_delta)
		{
			best_action_delta = action_delta;
			
			temp_pair_num = pair_city_num;			
			for(int j=0; j<2*pair_city_num; j++)
				temp_city_sequence[j] = city_sequence[j];			
		}
		
		if(best_action_delta > 0)	
			break;
	}	
	
	// restore the action with the best delta
	pair_city_num = temp_pair_num;
	for(int i=0; i<2*pair_city_num; i++)
		city_sequence[i] = temp_city_sequence[i];		

	return best_action_delta;	
}

//Execute the best action stored in city_sequence[] with depth pair_city_num
bool execute_best_action()
{
	int begin_city = city_sequence[0];
	int cur_city = city_sequence[1];	
	all_node[begin_city].next_city = NULL_1;
	all_node[cur_city].pre_city = NULL_1;	

	for(int i=1; i<pair_city_num; i++)  
	{
		int next_city_to_connect = city_sequence[2*i];
		int next_city_to_disconnect = city_sequence[2*i+1];	
		
		reverse_sub_path(cur_city, next_city_to_disconnect);		
				
		all_node[cur_city].next_city = next_city_to_connect;
		all_node[next_city_to_connect].pre_city = cur_city;
		all_node[next_city_to_disconnect].pre_city = NULL_1;
		
		cur_city=next_city_to_disconnect;				
	}
			
	all_node[begin_city].next_city=cur_city;
	all_node[cur_city].pre_city=begin_city;	
	
	if(check_solution_feasible()==false)
		return false;		

	return true;
}


// ---------------------------- MCTS  -------------------------- //
// process of the mcts
void mcts()
{	 
	//while(true)
	while(((double)clock() - begin_time) / CLOCKS_PER_SEC < (double)(time_limit))
	{
		int before_simulation_distance = get_solution_total_distance();
		
		//simulate a number of (controled by param_h) actions
		int best_delta = simulation(param_h*city_num);
		
		// Use the information of the best action to update the parameters of mcts by back propagation	
		back_propagation(before_simulation_distance, best_delta);
				
		if(best_delta > 0)
		{
			// select the best action to execute
			execute_best_action();							
			
			// store the best found solution to struct_Node *best_all_node				
			int cur_solution_total_distance = get_solution_total_distance();		
			if(cur_solution_total_distance < best_distance)
			{
				best_distance = cur_solution_total_distance;	
				store_best_solution();					
			}			
		}		
		else{
			if (continue_flag == 1){
				unsigned int seed = static_cast<unsigned int>(time(0));
				srand(seed);
			}
			else{
				break;
			}
		}
	}	
} 

#endif