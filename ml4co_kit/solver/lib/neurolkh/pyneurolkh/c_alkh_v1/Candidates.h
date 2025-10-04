#ifndef CANDIDATES_H
#define CANDIDATES_H

#include "ALKH.h"


int compareCandidates(const void *a, const void *b) 
{
    Candidate *c1 = (Candidate *)a;
    Candidate *c2 = (Candidate *)b;
    return (c1->cost > c2->cost) - (c1->cost < c2->cost);
}


void createCandidates(void)
{
    int node_id, _node_id, select_num;
    double max_cost;
    int max_cost_idx;
    Node *curNode = FirstNode;
    Node *_curNode;

    // set MAX_CANDIDATES for each Node
    do {
        curNode -> max_candidates = base_max_candidates * (abs(curNode -> degree) + 1);
        initNodeCandidates(curNode);
    } while((curNode = curNode -> suc) != FirstNode);

    do {
        // Init
        select_num = 0;
        node_id = curNode -> id;
        _curNode = FirstNode;
        max_cost_idx = -1;
        max_cost = INT_MIN;

        // Get Candidates
        do {
            _node_id = _curNode -> id;
    
            // Continue if the Node is the same
            if (_node_id == node_id) continue;

            // If select_num < MAX_CANDIDATES, then directly add it
            // Else, Compare it with Max Cost Candidate Node.
            struct Candidate cand = {_curNode, getCost(curNode, _curNode), INT_MAX};
            if (select_num < curNode->max_candidates) {
                curNode -> candidates[select_num++] = cand;

                // Update max_cost and max_cost_idx if necessary
                if (cand.cost > max_cost) {
                    max_cost = cand.cost;
                    max_cost_idx = select_num - 1;
                }
            }
            else {
                if (cand.cost < max_cost) {
                    curNode->candidates[max_cost_idx] = cand;
                    // Find new max_cost and max_cost_idx
                    max_cost = cand.cost;
                    for (int i = 0; i < curNode->max_candidates; ++i) {
                        if (curNode->candidates[i].cost > max_cost) {
                            max_cost = curNode->candidates[i].cost;
                            max_cost_idx = i;
                        }
                    }
                }
            }
        } while((_curNode = _curNode -> suc) != FirstNode);
        qsort(
            curNode->candidates, curNode->max_candidates, 
            sizeof(Candidate), compareCandidates
        );
    } while((curNode = curNode -> suc) != FirstNode);
}


void sortCandidates(void)
{
    Node *curNode = FirstNode;
    do {
        for(int i=0; i<base_max_candidates; i++){
            curNode->candidates[i].cost = getCost(curNode, curNode->candidates[i].To);
        }
        qsort(
            curNode->candidates, 
            curNode->max_candidates, 
            sizeof(Candidate), 
            compareCandidates
        );
    } while((curNode = curNode -> suc) != FirstNode);
}

void freeCandidates(void)
{
    Node *curNode = FirstNode;
    do {
        freeNodeCandidates(curNode);
    } while((curNode = curNode -> suc) != FirstNode);
}

#endif