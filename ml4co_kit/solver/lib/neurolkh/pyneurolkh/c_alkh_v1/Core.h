#ifndef CORE_H
#define CORE_H

#include "ALKH.h"


// Function to calculate the cost of the dense M1 tree
double denseM1Tree(void) 
{
    // Initialize parameters
    struct Node *Blue, *nextBlue, *curNode, *_curNode, *N1;
    Blue = FirstNode;
    double cost, min_cost;
    double max_min_cost = INT_MIN;
    double m1tree_sum_cost = 0;

    // Initialize node degrees
    curNode = FirstNode;

    do {
        curNode->degree = -2;
        curNode->exp = NULL;
        curNode->exp_cost = INT_MAX;
    } while ((curNode = curNode->suc) != FirstNode);

    // Main loop to construct the M1 tree
    while ((curNode = Blue->suc) != FirstNode) {
        min_cost = INT_MAX;

        // Find nextBlue and update all nodes' expected costs and nodes
        do {
            cost = getCost(Blue, curNode);
            if (cost < curNode->exp_cost) {
                curNode->exp_cost = cost;
                curNode->exp = Blue;
            }
            if (curNode->exp_cost < min_cost) {
                min_cost = curNode->exp_cost;
                nextBlue = curNode;
            }
        } while ((curNode = curNode->suc) != FirstNode);

        // Link nextBlue after Blue
        Follow(Blue, nextBlue);
        Blue = nextBlue;
    }

    // Update node degrees and calculate the total cost of the M1 tree
    curNode = FirstNode;
    while ((curNode = curNode->suc) != FirstNode){
        m1tree_sum_cost += curNode->exp_cost;
        m1tree_sum_cost -= 2 * curNode->pi;
        curNode->degree++;
        curNode->exp->degree++;
    }
    m1tree_sum_cost -= 2*FirstNode -> pi;

    // Set the expected node and cost for FirstNode
    FirstNode->exp = FirstNode->suc;
    FirstNode->exp_cost = FirstNode->suc->exp_cost;

    // If the node degree is -1, then find the subopt Node to connect
    curNode = FirstNode;
    do {
        if (curNode->degree == -1){
            curNode -> subopt_cost = INT_MAX;

            _curNode = FirstNode;
            do {
                if (curNode == _curNode) continue;
                if (curNode->exp == _curNode) continue;
                if (_curNode->exp == curNode) continue;
                
                cost = getCost(curNode, _curNode);
                if (cost < curNode -> subopt_cost){
                    curNode -> subopt_cost = cost;
                    curNode -> subopt = _curNode;
                }
                if (cost <= max_min_cost) break;

            } while((_curNode = _curNode->suc) != FirstNode);

            if (curNode->subopt_cost != INT_MAX && curNode->subopt_cost > max_min_cost){
                max_min_cost = curNode -> subopt_cost;
                N1 = curNode;
            }
        }
    } while ((curNode = curNode->suc) != FirstNode);

    // Update FirstNode
    N1 -> subopt -> degree++;
    N1 -> degree ++;

    m1tree_sum_cost += N1 -> subopt_cost;
    // if (N1 == FirstNode){
    //     N1 -> suc -> exp = 0;
    // }
    // else{
    //     N1 -> suc -> exp = 0;
    //     Precede(N1, FirstNode);
    //     FirstNode = N1;
    // }
    if (N1 != FirstNode){
        Precede(N1, FirstNode);
        FirstNode = N1;
    }
    
    // Update Norm
    curNode = FirstNode;
    Norm = 0;
    do {
        Norm += curNode->degree * curNode->degree;
    } while ((curNode = curNode->suc) != FirstNode);

    // Return the total cost of the M1 tree
    return m1tree_sum_cost;
}


// Function to calculate the cost of the sparse M1 tree
double sparseM1Tree(void) 
{
    // Initialize parameters
    struct Node *Blue, *nextBlue, *curNode, *N1;
    Blue = FirstNode;
    double cost;
    double m1tree_sum_cost = 0;
    double max_min_cost = INT_MIN;
    curNode = FirstNode;

    // Insert all nodes into the priority queue lazily, except FirstNode
    while ((curNode = curNode->suc) != FirstNode) {
        curNode->degree = -2;
        curNode->exp = Blue;
        curNode->exp_cost = INT_MAX;
        pq_lazy_insert(curNode);
    }
    FirstNode->degree = -2;

    // Update all candidate nodes to the Blue node
    for (int i = 0; i < Blue->max_candidates; ++i) {
        curNode = Blue->candidates[i].To;
        cost = getCost(Blue, curNode);

        // Update node information
        curNode->exp = Blue;
        curNode->exp_cost = cost;

        // Since the node information has changed, update the priority queue
        pq_heapify_up(curNode->index);
    }

    // Main loop to construct the M1 tree
    for (int i = 1; i < nodes_num; ++i) {
        // Extract the next Blue node from the priority queue and link it after the current Blue node
        nextBlue = pq_extract_min();
        Follow(Blue, nextBlue);
        Blue = nextBlue;

        // Update all candidate nodes to the Blue node
        for (int i = 0; i < Blue->max_candidates; ++i) {
            curNode = Blue->candidates[i].To;

            // Skip if the node is already in the MST
            if (curNode->index == -1) continue;

            // Get cost
            cost = getCost(Blue, curNode);

            // Update node information if the new cost is lower
            if (cost < curNode->exp_cost) {
                curNode->exp = Blue;
                curNode->exp_cost = cost;
                pq_heapify_up(curNode->index);
            }
        }
    }

    // Update node degrees and Caculate Sum Cost
    curNode = FirstNode;
    while ((curNode = curNode->suc) != FirstNode)
    {
        curNode -> degree++;
        curNode -> exp -> degree++;
        m1tree_sum_cost += curNode -> exp_cost;
        m1tree_sum_cost -= 2 * curNode -> pi;
        curNode -> subopt = 0;
    }
    m1tree_sum_cost -= 2 * FirstNode -> pi;

    // Set the expected node and cost for FirstNode
    FirstNode->exp = FirstNode->suc;
    FirstNode->exp_cost = FirstNode->suc->exp_cost;

    // If the node degree is -1, then find the subopt Node to connect
    curNode = FirstNode;
    do {
        if (curNode->degree == -1){
            curNode -> subopt_cost = INT_MAX;
            for(int i=0; i<curNode->max_candidates; ++i){
                // check there is not an exist edge between them 
                if (curNode->candidates[i].To == curNode->exp) continue;
                if (curNode->candidates[i].To->exp == curNode) continue;

                cost = getCost(curNode, curNode->candidates[i].To);
                if (cost < curNode -> subopt_cost){
                    curNode -> subopt_cost = cost;
                    curNode -> subopt = curNode -> candidates[i].To;
                }
                if (cost <= max_min_cost) break;
            }

            if (curNode->subopt_cost != INT_MAX && curNode->subopt_cost > max_min_cost){
                max_min_cost = curNode -> subopt_cost;
                N1 = curNode;
            }
        }
    } while ((curNode = curNode->suc) != FirstNode);

    // Update FirstNode
    N1 -> subopt -> degree++;
    N1 -> degree ++;

    m1tree_sum_cost += N1 -> subopt_cost;
    if (N1 != FirstNode){
        Precede(N1, FirstNode);
        FirstNode = N1;
    }

    // Update Norm
    curNode = FirstNode;
    Norm = 0;
    do {
        Norm += curNode->degree * curNode->degree;
    } while ((curNode = curNode->suc) != FirstNode);

    // Return the total cost of the M1 tree
    return m1tree_sum_cost;
}


double Ascent(void) 
{
    // Prepare Some Params
    struct Node *t;
    double BestW, W, W0;
    int T, Period, P, InitialPhase, BestNorm;
    int Precision, InitialStepSize;
        
    // Dense M1Tree
    W = denseM1Tree();

    // Create Candidates
    createCandidates();

    // Set Node last_degree as degree
    t = FirstNode;
    do
        t->last_degree = t->degree;
    while ((t = t->suc) != FirstNode);

    // Main Loop
    BestW = W0 = W;
    BestNorm = Norm;
    InitialPhase = 1;
    Precision = 100;
    InitialStepSize = 1;

    for(Period = InitialPeriod, T = InitialStepSize * Precision;
        Period > 0 && T > 0 && Norm != 0; Period /= 2, T /= 2) {
        for (P = 1; T && P <= Period && Norm != 0; P++) {
            // Adjust the Pi-values
            t = FirstNode;
            do {
                if (t->degree != 0) {
                    t->pi += LR * T * (7 * t->degree + 3 * t->last_degree) / 10;
                    if (t->pi > INT_MAX / 1000)
                        t->pi = INT_MAX / 1000;
                    else if (t->pi < INT_MIN / 1000)
                        t->pi = INT_MIN / 1000;
                }
                t->last_degree = t->degree;
            }
            while ((t = t->suc) != FirstNode);

            // Compute a minimum 1-tree in the sparse graph
            W = sparseM1Tree();

            // std::cout << "P = " << P << ", T = " << T << ", W = " << W;
            // std::cout << ", BestW = "<< BestW << ", Norm = " << Norm;
            // std::cout << ", BestNorm = " << BestNorm << ", Period = ";
            // std::cout << Period << std::endl;

            // Check if an improvement has been found
            if (W > BestW || (W == BestW && Norm < BestNorm)) {
                // Update BestW and BestNorm
                BestW = W;
                BestNorm = Norm;

                // Update the BestPi-values
                t = FirstNode;
                do
                    t->best_pi = t->pi;
                while ((t = t->suc) != FirstNode);

                // If in the initial phase, the step size is doubled
                if (InitialPhase && T * sqrt((double) Norm) > 0)
                    T *= 2;

                // If the improvement was found at the last iteration of the 
                // current period, then double the period
                if (P == Period && (Period *= 2) > InitialPeriod)
                    Period = InitialPeriod;
            } 
            else {
                if (InitialPhase && P > Period / 2) {
                    // Conclude the initial phase
                    InitialPhase = 0;
                    P = 0;
                    T = 3 * T / 4;
                }     
            }
        }
    }

    // Final Update
    t = FirstNode;
    do{
        t->pi = t->best_pi;
        t->best_pi = 0;
    } while ((t = t->suc) != FirstNode);

    return W;
}

#endif
