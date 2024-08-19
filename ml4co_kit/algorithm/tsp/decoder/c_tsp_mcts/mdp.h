#ifndef MDP_H
#define MDP_H

#include "tsp.h"


int mdp()
{
	mcts_init();                      // Initialize MCTS parameters
	generate_initial_solution();      // State initialization of MDP	
	local_search_by_2opt_move();	  // 2-opt based local search within small neighborhood	
	mcts();		                      // Tageted sampling via MCTS within enlarged neighborhood

	// Repeat the following process until termination
	while(((double)clock() - begin_time) / CLOCKS_PER_SEC < (double)(time_limit))
	{	
		generate_initial_solution();				
		local_search_by_2opt_move();				
		mcts();	
	}
	// Copy information of the best found solution (stored in Struct_Node *Best_All_Node ) to Struct_Node *All_Node 
	restore_best_solution();
	
	if(check_solution_feasible())
		return get_solution_total_distance();
	else
		return INF;
}

#endif
