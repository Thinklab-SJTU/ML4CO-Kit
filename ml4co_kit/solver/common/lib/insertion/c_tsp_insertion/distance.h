#ifndef DISTANCE_H
#define DISTANCE_H

#include "insertion.h"

double get_distance(int node1_idx, int node2_idx)
{
	double x = coord_x[node1_idx] - coord_x[node2_idx];
	double y = coord_y[node1_idx] - coord_y[node2_idx];
	double dist = sqrt(x*x + y*y);
  	return dist;
}


double get_insert_distance(int node1_idx, int node2_idx, int insert_node_idx)
{
	double cut_distance = get_distance(node1_idx, node2_idx);
	double add_distance_1 = get_distance(node1_idx, insert_node_idx);
	double add_distance_2 = get_distance(node2_idx, insert_node_idx);
	double insert_distance = add_distance_1 + add_distance_2 - cut_distance;
	return insert_distance;
}

#endif  // DISTANCE_H