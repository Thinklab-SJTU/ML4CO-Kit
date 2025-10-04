#ifndef MEMORY_H
#define MEMORY_H

#include "ALKH.h"


void allocateMemory(){
    outputs = new double[(out_candidates_num+2)*nodes_num];
}


void releaseMemory(){
    freeNodeList(FirstNode);
}


#endif