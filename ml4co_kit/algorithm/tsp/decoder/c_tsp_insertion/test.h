#ifndef TEST_H
#define TEST_H

#include "insertion.h"

// print number
void print_num(int x)
{
    std::cout << x << std::endl;
}

// print solution
void print_solution(void)
{
    for(int i=0; i<=end_idx; ++i){
        std::cout << solution[i] << " ";
        if (i % 10 == 9){
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void print_solution_length(void)
{
    double length = 0;
    for(int i=0; i<nodes_num; ++i){
        length = length + get_distance(solution[i], solution[i+1]);
    }
    std::cout << "length:" << length << std::endl;
}

#endif