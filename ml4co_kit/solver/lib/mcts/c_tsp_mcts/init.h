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
} 


//Estimate the potential of each edge by upper bound confidence function
double temp_get_potential(int city_1, int city_2)
{	
	return pow(2.718, 1*weight[city_1][city_2]);	
}


// Indentify the promising cities as candidates which are possible to connect to cur_city
void temp_identify_promising_city()
{
	promising_city_num=0;
	for(int i=0;i<city_num;i++)	
	{
		if(if_city_selected[i]==true)
			continue;
		promising_city[promising_city_num++]=i;	
	}
}


// Set the probability (stored in probabilistic[]) of selecting each candidate city (proportion to the potential of the corresponding edge)
bool temp_get_probabilistic(int cur_city)
{
	if(promising_city_num==0)
		return false;
		
	double total_potential=0;
	for(int i=0; i<promising_city_num; i++)	
		total_potential += temp_get_potential(cur_city, promising_city[i]);	
		
	probabilistic[0]=(int)(1000 * temp_get_potential(cur_city, promising_city[0]) / total_potential);
	for(int i=1; i<promising_city_num-1; i++)
		probabilistic[i] = probabilistic[i-1] + (int)(1000 * temp_get_potential(cur_city, promising_city[i]) / total_potential);	
	probabilistic[promising_city_num-1] = 1000;	

	return true;
}


// probabilistically choose a city, controled by the values stored in probabilistic[] 
int temp_probabilistic_get_city_to_connect()
{
	int Random_Num = get_random_num(1000);
	for(int i=0; i<promising_city_num; i++)
		if(Random_Num < probabilistic[i])
			return promising_city[i];
	
	return NULL_1;
}


// The whole process of choosing a city (a_{i+1} in the paper) to connect cur_city (b_i in the paper)
int temp_choose_city_to_connect(int cur_city)
{	
	temp_identify_promising_city();
	temp_get_probabilistic(cur_city);	
	
	return temp_probabilistic_get_city_to_connect();
}


// Generate Initial Solution
bool generate_initial_solution()
{
	int selected_city_num=0;
	int cur_city=start_city;
	int next_city;	

	for(int i=0;i<city_num;i++)	{
		solution[i] = NULL_1;
		if_city_selected[i]=false;		
	}
	
	solution[selected_city_num++]=cur_city;
	if_city_selected[cur_city]=true;
	do{ 	
		next_city = temp_choose_city_to_connect(cur_city);	
		if(next_city != NULL_1){
			solution[selected_city_num++]=next_city;
			if_city_selected[next_city]=true;						
			cur_city = next_city;			
		}		
	} while(next_city != NULL_1);

	convert_solution_to_all_node();
	
	if(check_solution_feasible()==false)
		return false;
	
	return true;		
}

#endif