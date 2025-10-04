#ifndef Priority_Queue_H
#define Priority_Queue_H

#include "ALKH.h"

// Swap function
void swap(struct Node **a, struct Node **b) {
    struct Node *temp = *a;
    *a = *b;
    *b = temp;
    (*a)->index = a - pq;
    (*b)->index = b - pq;
}

// Insert node into priority queue
void pq_insert(struct Node *node) {
    node->index = pq_size;
    pq[pq_size] = node;
    pq_size++;
    pq_heapify_up(node->index);
}

// Lazy insert node into buffer
void pq_lazy_insert(struct Node *node) {
    node->index = pq_size;
    pq[pq_size] = node;
    pq_size++;
}

// Extract the node with minimum pi value
struct Node *pq_extract_min() {
    if (pq_size == 0) {
        return NULL;
    }
    struct Node *min_node = pq[0];
    pq[0] = pq[pq_size - 1];
    pq[0]->index = 0;
    pq_size--;
    pq_heapify_down(0);
    min_node->index = -1;
    return min_node;
}

// Heapify up
void pq_heapify_up(int i) {
    while (i > 0 && pq[(i - 1) / 2]->exp_cost > pq[i]->exp_cost) {
        swap(&pq[(i - 1) / 2], &pq[i]);
        i = (i - 1) / 2;
    }
}

// Heapify down
void pq_heapify_down(int i) {
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    int smallest = i;

    if (left < pq_size && pq[left]->exp_cost < pq[smallest]->exp_cost) {
        smallest = left;
    }
    if (right < pq_size && pq[right]->exp_cost < pq[smallest]->exp_cost) {
        smallest = right;
    }
    if (smallest != i) {
        swap(&pq[i], &pq[smallest]);
        pq_heapify_down(smallest);
    }
}

// Initialize priority queue
void pq_init(int max_size) {
    pq = (struct Node **)malloc(max_size * sizeof(struct Node *));
    pq_size = 0;
}

// Free priority queue
void pq_free() {
    free(pq);
}

#endif