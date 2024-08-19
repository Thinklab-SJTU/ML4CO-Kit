#ifndef MEMORY_H
#define MEMORY_H

#include "insertion.h"

// Allocate Memory
void allocate_memory(int nodes_num)
{
	// input parameters
	coord_x = new double [nodes_num];  
	coord_y = new double [nodes_num];  
	order = new int [nodes_num];
	solution = new int [nodes_num+1];
}


// Release Memory
void release_memory(int nodes_num)
{
	delete []coord_x;  
	delete []coord_y;  
	delete []order;
} 

#endif