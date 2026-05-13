#ifndef CANDIDATES_H
#define CANDIDATES_H

#include "ALKH.h"
#include <algorithm> 


int compareCandidates(const void *a, const void *b) 
{
    Candidate *c1 = (Candidate *)a;
    Candidate *c2 = (Candidate *)b;
    return (c1->cost > c2->cost) - (c1->cost < c2->cost);
}


void createCandidates(int *_candidates)
{
    int node_id, next_idx, select_num;
    int tmp_array[base_max_candidates];
    Node *curNode = FirstNode;
    Node *_curNode;

    // set MAX_CANDIDATES for each Node
    do {
        curNode -> max_candidates = base_max_candidates;
        initNodeCandidates(curNode);
    } while((curNode = curNode -> suc) != FirstNode);
    
    do {
        // Init
        select_num = 0;
        next_idx = 0;
        node_id = curNode -> id;
        _curNode = FirstNode;
        
        // Get Candidates
        for(int i=0; i<base_max_candidates; ++i){
            tmp_array[i] = _candidates[node_id*base_max_candidates+i];
        }
        std::sort(tmp_array, tmp_array + base_max_candidates);

        // Create Candidates for Current Node
        do {
            if (_curNode -> id != tmp_array[next_idx]) continue;
            next_idx++;
            struct Candidate cand = {_curNode, getCost(curNode, _curNode), INT_MAX};
            curNode -> candidates[select_num++] = cand;
        } while((_curNode = _curNode -> suc) != FirstNode);

        // Sort from low to high
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