#include<stdio.h>
#include<stdlib.h>
#include<math.h>


void check_data(double * dist, int n) {
    for (int i = 0; i < n * n; i++) {
        if (!isnormal(dist[i]) && dist[i]!=0.) {
            printf("Error: the data is invalid! [%d]-%lf\n", i, dist[i]);
            exit(1);
        }
    }
}
            
int nearest(int last, int n, double* dist, int* node_flag){
    double cur_min_dist = 1e8;
    int res;
    for (int j=0; j<n; j++){ 
        // try node j
        if (node_flag[j]) continue;
        // from node last -> j
        if (dist[last * n + j] < cur_min_dist) {
            cur_min_dist = dist[last * n + j];
            res = j;
        }
    }
    return res;
}


void nearest_neighbor(int n, double * dist, int * path, double * cost) {
    check_data(dist, n);
    
    int * node_flag = (int *)malloc(sizeof(int) * n); // recording whether a node is visited
    if (node_flag == NULL) {printf("Error malloc.\n"); exit(0);}

    double cur_min_dist = 1e8;
    int start = 0, last, cur;

    for (int i=0; i<n; i++) node_flag[i] = 0;
    last = start;
    node_flag[last] = 1;
    path[0] = last;
    *cost = 0;

    // search from node start
    for (int step=1; step<n; step++) { // try n-1 steps;
        last = path[step] = nearest(last, n, dist, node_flag);
        node_flag[last] = 1;
    }

    // record the solution
    for (int step=0; step<n; step++) *cost += dist[path[step] * n + path[(step + 1) % n]];

    free(node_flag);
}
