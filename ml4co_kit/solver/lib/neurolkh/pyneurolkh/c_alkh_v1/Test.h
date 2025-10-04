#ifndef TEST_H
#define TEST_H

#include "ALKH.h"

void print_number(int num){
    std::cout << num << std::endl;
}

void print_nodes_id(void)
{
    int i = 0;
    struct Node *curNode = FirstNode;
    std::cout << "Nodes ID:"<< std::endl;
    do {
        std::cout << "(" << curNode->pred->id << ", " << curNode->id << ", " << curNode->suc->id << "), ";
    } while ((curNode = curNode->suc) != FirstNode and i++ < nodes_num);
    std::cout << std::endl;
}


void print_nodes_info(void)
{
    int i = 0;
    struct Node *curNode = FirstNode;
    std::cout << "Nodes Info:"<< std::endl;
    do {
        std::cout << curNode->id << ": x(" << curNode->x << "), y(" << curNode->y;
        std::cout << "), index(" << curNode->index << "), pi(" << curNode->pi;
        std::cout << "), best_pi(" << curNode->best_pi << "), degree(" << curNode->degree;
        std::cout << "), pred(" << curNode->pred->id << "), suc(" << curNode->suc->id;
        if (curNode -> exp)
            std::cout << "), exp(" << curNode->exp->id << "), exp_cost(" << curNode->exp_cost;
        else
            std::cout << "), exp(NULL), exp_cost(NULL";
        if (curNode -> subopt)
            std::cout << "), subopt(" << curNode->subopt->id << "), subopt_cost(" << curNode->subopt_cost;
        else
            std::cout << "), subopt(NULL), subopt_cost(NULL";
        std::cout << ")" <<std::endl;
    } while ((curNode = curNode->suc) != FirstNode and i++ < nodes_num);
}


void print_candidates(void)
{
    int i = 0;
    struct Node *curNode = FirstNode;
    std::cout << "Candidates:"<< std::endl;
    do {
        std::cout << curNode->id << ": ";
        for(int j=0; j<curNode->max_candidates; ++j){
            std::cout << "(" << curNode->candidates[j].To->id << ", " << curNode->candidates[j].cost << ") ";
        }
        std::cout << std::endl;
    } while ((curNode = curNode->suc) != FirstNode and i++ < nodes_num);
    std::cout << std::endl;
}

void print_nodes_degrees(void)
{
    int i = 0;
    struct Node *curNode = FirstNode;
    std::cout << "Nodes Degree:"<< std::endl;
    do {
        std::cout << curNode->degree << " ";
    } while ((curNode = curNode->suc) != FirstNode and i++ < nodes_num);
    std::cout << std::endl;
}

#endif