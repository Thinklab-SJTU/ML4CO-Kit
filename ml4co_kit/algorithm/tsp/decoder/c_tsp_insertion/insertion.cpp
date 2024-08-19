#include "insertion.h"
#include "input.h"
#include "memory.h"
#include "greedy.h"
#include "distance.h"
#include "test.h"
#include <iostream>


extern "C" {
	int* insertion(
		short* input_order, 
		float *nodes_coords, 
		int input_nodes_num
	){
		nodes_num = input_nodes_num;
        allocate_memory(input_nodes_num);
        read_nodes_coords(nodes_coords);
        read_order(input_order);
        greedy_insertion();
        release_memory(input_nodes_num);
		return solution;	
	}
}
