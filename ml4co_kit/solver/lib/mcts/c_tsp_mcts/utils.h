#ifndef UTILS_H
#define UTILS_H
#include "tsp.h"

// ----------------------------- RANDOM ----------------------------- //

int get_random_num(int range)
{ 
	return rand() % range;
}


// ----------------------- CALCULATE DISTANCE ----------------------- // 

double calculate_double_distance(int city_1, int city_2)
{
	int x = coord_x[city_1] - coord_x[city_2];
	int y = coord_y[city_1] - coord_y[city_2];
	double dist = sqrt(x*x + y*y);
  	return dist;
}

int calculate_int_distance(int city_1, int city_2)
{	
  	return (int)(0.5 + calculate_double_distance(city_1, city_2));
}

void calculate_all_pair_distance()
{
  	for(int i=0; i<city_num; i++)
  		for(int j=0; j<city_num; j++)
  		{
	  		if(i!=j)
	    		distance[i][j] = calculate_int_distance(i,j);  
	  		else
	    		distance[i][j] = INF;  	  
		}  	 
}

int get_distance(int city_1,int city_2)
{
	return distance[city_1][city_2];
}

int get_solution_total_distance()
{
  	int solution_total_distance = 0;
  	for(int i=0;i<city_num;i++)
  	{
  		int tmp_next = all_node[i].next_city;
  		if(tmp_next != NULL_1)
  	  		solution_total_distance += get_distance(i, tmp_next); 
  		else
  			return INF; 	  		
  	}	
  
  	return solution_total_distance;
}

double get_current_solution_double_distance()
{
  	double current_solution_double_distance=0;
  	for(int i=0;i<city_num;i++)
  	{
  		int tmp_next=all_node[i].next_city;
  		if(tmp_next != NULL_1)
  	  		current_solution_double_distance += calculate_double_distance(i, tmp_next);
  		else
  			return INF;
  	}	
  	return current_solution_double_distance;
}


// --------------------------- CONVERSION --------------------------- //  

void convert_solution_to_all_node()
{
  	int tmp_cur;
  	int tmp_pre;
  	int tmp_next;
  
  	for(int i=0; i<city_num; i++)
  	{
  		tmp_cur = solution[i];  
  		tmp_pre = solution[(i-1+city_num) % city_num];
  		tmp_next = solution[(i+1+city_num) % city_num];  		
  		all_node[tmp_cur].pre_city = tmp_pre;
  		all_node[tmp_cur].next_city = tmp_next;   
  	} 
}
 
bool convert_all_node_to_solution()
{
	for(int i=0; i<city_num; i++)
		solution[i] = NULL_1;
	
	int cur_index = 0;
	int cur_city = start_city;
	solution[cur_index] = start_city;
	
	do
	{
		cur_index ++;
		cur_city = all_node[cur_city].next_city;
		if(cur_city == NULL_1 || cur_index >= city_num)
			return false;
		solution[cur_index] = cur_city;			
	} while(all_node[cur_city].next_city != start_city);
	
	return true;
}


// ----------------------------------- CHECK ---------------------------------- //  

bool check_solution_feasible()
{
	int cur_city=start_city;
	int visited_city_num=0;
	while(true)
	{	
		cur_city = all_node[cur_city].next_city;
		visited_city_num ++;

		if(cur_city == NULL_1 || visited_city_num > city_num) 	
			return false;
		
		if(cur_city == start_city && visited_city_num == city_num)	
			return true;			 	
	}
}

bool check_if_two_city_same_or_adjacent(int city_1, int city_2)
{
	if(city_1==city_2 || all_node[city_1].next_city == city_2 || all_node[city_2].next_city == city_1)	
		return true;
	else
		return false;
}

// ----------------------------- SELECT & CANDIDATE ----------------------------- //  

int get_best_unselected_city(int cur_city)
{	
	int best_unselected = NULL_1;
	for(int i=0; i<city_num; i++)
	{
		if(i==cur_city || if_city_selected[i] || get_distance(cur_city,i) >= INF )
			continue;
			
		if(best_unselected == NULL_1 || edge_heatmap[cur_city][i] > edge_heatmap[cur_city][best_unselected])
			best_unselected = i;
	}
	
	if(edge_heatmap[cur_city][best_unselected] >= 0.0001)	
		return best_unselected;
	else
		return NULL_1;
}

void identify_candidate_set()
{	
	for(int i=0; i<city_num; i++)
	{
		candidate_num[i] = 0;
		for(int j=0;j<city_num;j++)
			if_city_selected[j]=false;	
		while(true)	
		{
			int best_unselected_city = get_best_unselected_city(i);
			if(best_unselected_city != NULL_1)
			{
				candidate[i][candidate_num[i]++] = best_unselected_city;		
				if_city_selected[best_unselected_city] = true;
			}
			else
				break;			
		}	
	}
}


// ------------------------------ STORE & RESTORE ------------------------------- //  

void store_best_solution()
{
	for(int i=0;i<city_num;i++)
	{
		best_all_node[i].next_city=all_node[i].next_city;
		best_all_node[i].pre_city=all_node[i].pre_city;
	}	 
}

void restore_best_solution()
{
	for(int i=0;i<city_num;i++)
	{
		all_node[i].next_city = best_all_node[i].next_city;
		all_node[i].pre_city = best_all_node[i].pre_city;
	}	 
}


// ---------------------------------- REVERSE ----------------------------------- //  

void reverse_sub_path(int city_1,int city_2)
{
	int cur_city=city_1;
	int tmp_next=all_node[cur_city].next_city;
	
	while(true)
	{	
		int tmp_city = all_node[cur_city].pre_city;
		all_node[cur_city].pre_city = all_node[cur_city].next_city;
		all_node[cur_city].next_city = tmp_city;
		
		if(cur_city==city_2)
			break;

		cur_city=tmp_next;
		tmp_next=all_node[cur_city].next_city;	
	}	
} 

#endif 