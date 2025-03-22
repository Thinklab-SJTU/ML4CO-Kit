#include "Individual.h"
#include "LocalSearch.h"
#include "Params.h"
#include "CircleSector.h"
#include "Split.h"
using namespace std;


extern "C" {
	int* cvrp_local_search(
		short* tour,
		float *nodes_coords,
		float *demands,
		int nodes_num,
		int tour_length,
		int coords_scale,
		int demands_scale,
		int seed
	){
		// Problem Params
		Params params(nodes_coords, demands, nodes_num, coords_scale, demands_scale, seed);

		// Input Initial Tours
		Individual indiv(&params, tour, tour_length);
		Split split(&params);

		// Creating Local Search
		LocalSearch localSearch(&params, &split);

		// Local Search!
		localSearch.run(&indiv, params.penaltyCapacity, params.penaltyDuration);

		// Return
		int *ls_tours = new int[tour_length + 2];
		if (indiv.isFeasible) {
			ls_tours[0] = 0;
			int ls_idx = 1;
			for (size_t i = 0; i < indiv.chromR.size(); ++i){
				if (!indiv.chromR[i].empty()) {
					for (int node : indiv.chromR[i]) {
						ls_tours[ls_idx] = node;
						ls_idx ++;
					}
					ls_tours[ls_idx] = 0;
					ls_idx ++;
				}
			}
			ls_tours[ls_idx] = -1;
		}
		else {
			ls_tours[0] = -1;
		}
		return ls_tours;
	}
}