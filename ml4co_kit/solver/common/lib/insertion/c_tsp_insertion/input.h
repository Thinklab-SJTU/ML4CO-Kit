#ifndef INPUT_H
#define INPUT_H

#include "insertion.h"

void read_nodes_coords(float *nodes_coords)
{	
	int i;
	for(i=0; i<nodes_num; i++)
	{
		coord_x[i]= nodes_coords[2*i];
		coord_y[i]= nodes_coords[2*i+1];			
	}
}


void read_order(short* input_order){
    int i;
	for(i=0; i<nodes_num; i++)
	{
		order[i] = input_order[i];			
	}
}


#endif  // INPUT_H