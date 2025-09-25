#ifndef INIT_H
#define INIT_H

#include "tsp.h"

// Allocate Memory
void allocate_memory(int city_num)
{
	// input parameters
	coord_x = new double [city_num];  
	coord_y = new double [city_num];  
	distance = new int *[city_num];
	for(int i=0; i<city_num; i++)
		distance[i] = new int [city_num];  
	edge_heatmap = new double *[city_num];
	for(int i=0; i<city_num; i++)
		edge_heatmap[i] = new double [city_num]; 

	// city_info
	all_node = new struct Node [city_num];   	
	best_all_node = new struct Node [city_num]; 
	candidate_num = new int [city_num];
	candidate = new int *[city_num];
	for(int i=0; i<city_num; i++)
		candidate[i] = new int [MAX_CANDIDATE_NUM];  
	if_city_selected = new bool [city_num];	

	// mcts
	weight=new double *[city_num];
	for(int i=0; i<city_num; i++)
		weight[i] = new double [city_num];  
	chosen_times = new int *[city_num];
	for(int i=0; i<city_num; i++)
		chosen_times[i] = new int [city_num];
	promising_city = new int [city_num];
	probabilistic = new int [city_num];	
	city_sequence=new int [city_num];
	temp_city_sequence = new int [city_num];
	gain = new int [2*city_num];
	real_gain = new int [2*city_num];
	
	// solution
	solution = new int [city_num];
	cur_solution = new int [city_num];
}


// Release Memory
void release_memory(int city_num)
{
	// input parameters
	delete []coord_x;  
	delete []coord_y;  
	for(int i=0; i<city_num; i++)
		delete []distance[i];  
	delete []distance;
	for(int i=0; i<city_num; i++)
		delete []edge_heatmap[i];
	delete []edge_heatmap;

	// city_info 
	delete []all_node;   
	delete []best_all_node;	
	delete []candidate_num;
	delete []if_city_selected;
	for(int i=0; i<city_num; i++)
		delete []candidate[i];
	delete []candidate;

	// mcts
	for(int i=0; i<city_num; i++)
		delete []weight[i];
	delete []weight;	
	for(int i=0; i<city_num; i++)
		delete []chosen_times[i];
	delete []chosen_times;	
	delete []promising_city;
	delete []probabilistic;	
	delete []city_sequence;
	delete []temp_city_sequence;
	delete []gain;
	delete []real_gain; 
	delete []cur_solution;
} 

#endif