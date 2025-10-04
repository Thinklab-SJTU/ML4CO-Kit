#ifndef CREATE_NODE_H
#define CREATE_NODE_H

#include "ALKH.h"

struct Node* createNode(int id,  float x, float y, float penalty)
{
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    if (!newNode) {
        perror("Failed to allocate memory for new node");
        exit(EXIT_FAILURE);
    }
    newNode->x = x;
    newNode->y = y;
    newNode->id = id;
    newNode->index = -1;
    newNode->max_candidates = 0;
    newNode->pi = (double)(penalty);
    newNode->best_pi = 0.0;
    newNode->degree = 0;
    newNode->pred = NULL;
    newNode->suc = NULL;
    newNode->exp = NULL;
    newNode->subopt = NULL;
    newNode->exp_cost = INT_MAX;
    newNode->real_cost = INT_MAX;
    newNode->subopt_cost = INT_MAX;
    newNode->candidates = NULL;
    return newNode;
}


void initNodeCandidates(struct Node* node) {
    node->candidates = (struct Candidate*)malloc(node -> max_candidates * sizeof(struct Candidate));
    if (node->candidates == NULL) {
        std::cerr << "Memory allocation for candidates failed!" << std::endl;
    }
}


void freeNodeCandidates(struct Node* node) {
    if (node->candidates != NULL) {
        free(node->candidates);
        node->candidates = NULL;
    }
}


double getCost(struct Node *a, struct Node *b)
{
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    return sqrt(dx * dx + dy * dy) + a->pi + b->pi;
}


void freeNode(struct Node* node)
{
    if (node != NULL) {
        free(node);
        node = NULL;
    }
}


void freeNodeList(struct Node* FirstNode) {
    if (FirstNode == NULL) {
        return;
    }

    struct Node* current = FirstNode;
    struct Node* nextNode;

    while (current->suc != FirstNode) {
        current = current->suc;
    }

    do {
        nextNode = current->suc;
        freeNode(current);
        current = nextNode;
    } while (current != FirstNode);

    freeNode(FirstNode);
}


void Follow(struct Node *a, struct Node *b)
{
    // If b is already follow a, then directly return
    if (a -> suc == b) return;

    // Params
    Node *a_suc = a -> suc;
    Node *b_pred = b -> pred;
    Node *b_suc = b -> suc;

    // Node B first detaches from the sequence it originally belongs to.
    b_pred -> suc = b_suc;
    b_suc -> pred = b_pred;

    // Node B is then inserted behind Node A.
    a -> suc = b;
    b -> pred = a;
    b -> suc = a_suc;
    a_suc -> pred = b;
}


void Precede(struct Node *a, struct Node *b)
{
    // If a is already precede b, then directly return
    if (a -> suc == b) return;

    // Params
    Node *a_suc = a -> suc;
    Node *a_pred = a -> pred;
    Node *b_pred = b -> pred;

    // Node A first detaches from the sequence it originally belongs to.
    a_pred -> suc = a_suc;
    a_suc -> pred = a_pred;

    // Node A is then inserted defore Node B.
    a -> pred = b_pred;
    b_pred -> suc = a;
    a -> suc = b;
    b -> pred = a;
}

#endif