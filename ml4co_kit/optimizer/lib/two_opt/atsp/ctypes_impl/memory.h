#ifndef MEMORY_H
#define MEMORY_H

#include "atsp.h"

// Allocate Memory
void allocate_memory(int city_num)
{
	// input parameters 
	distance = new int *[city_num];
	for(int i=0; i<city_num; i++)
		distance[i] = new int [city_num];  

	// city_info
	all_node = new struct Node [city_num];   	

	// solution
	solution = new int [city_num];
}


// Release Memory
void release_memory(int city_num)
{
	// input parameters  
	for(int i=0; i<city_num; i++)
		delete []distance[i];  
	delete []distance;

	// city_info 
	delete []all_node;
} 

#endif