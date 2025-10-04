#ifndef TEST_H
#define TEST_H

#include "tsp.h"


void print_cur_solution()
{
	for(int i=0; i<city_num; i++)
		cur_solution[i] = NULL_1;
	
	int cur_index = 0;
	int cur_city = start_city;
	cur_solution[cur_index] = start_city;
	
	do
	{
		cur_index ++;
		cur_city = all_node[cur_city].next_city;
		cur_solution[cur_index] = cur_city;	
	} while(all_node[cur_city].next_city != start_city);
	
    int i;
	for(i=0; i<city_num; i++){
		std::cout << cur_solution[i] << " ";
        if((i % 10) == 0) std::cout << std::endl;
    }
}

#endif