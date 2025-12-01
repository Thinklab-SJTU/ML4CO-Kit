#ifndef UTILS_H
#define UTILS_H
#include "atsp.h"

// ----------------------------- RANDOM ----------------------------- //

int get_random_num(int range)
{ 
	return rand() % range;
}


// ----------------------- CALCULATE DISTANCE ----------------------- // 

int get_distance(int city_1,int city_2)
{
	return distance[city_1][city_2];
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

bool check_if_two_city_same_or_adjacent(int city_1, int city_2)
{
	if(city_1==city_2 || all_node[city_1].next_city == city_2 || all_node[city_2].next_city == city_1)	
		return true;
	else
		return false;
}

#endif 