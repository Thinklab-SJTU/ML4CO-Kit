#ifndef READ_H
#define READ_H

#include "atsp.h"


void read_distance(float *dists)
{	
	int i;
	start_city = 0;
	for(i=0; i<city_num; i++){
    	for(int j=0; j<city_num; j++){	
	    	distance[i][j] = dists[i*city_num + j] * MAGNITY_RATE;
        }	
	}
}


void read_initial_solution(short *tour)
{
	int i;
	for (i=0; i<city_num; ++i)
		solution[i] = tour[i];
	convert_solution_to_all_node();	
}

#endif