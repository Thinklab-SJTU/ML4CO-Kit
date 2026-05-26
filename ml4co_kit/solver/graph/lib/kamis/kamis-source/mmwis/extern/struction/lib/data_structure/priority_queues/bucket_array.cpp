/******************************************************************************
 * bucket_array.cpp
 *
 * Copyright (C) 2015-2017 Sebastian Lamm <lamm@ira.uka.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "bucket_array.h"

#include "random_functions.h"

bucket_array::bucket_array(unsigned int number_of_nodes) {
    size = number_of_nodes;
    init();
}

bucket_array::~bucket_array() {
    delete [] position;
    delete [] array;
    delete [] value;
    delete [] first;
}

void bucket_array::init() {
    position = new unsigned int[size]; 
    array = new NodeID[size];
    value = new int[size];
    first = new unsigned int[size + 1];

    for (unsigned int i = 0; i < size; ++i) {
        position[i] = i;
        array[i] = i;
        value[i] = 0;
        first[i] = size;
    }
    first[0] = 0;
    first[size] = size;
    non_negative = 0;
}

void bucket_array::increment(NodeID node, int value) {
    for (; value > 0; --value) {
        increment(node);
    }
}

void bucket_array::increment(NodeID node) {
    int node_value = value[node];
    int last = first[node_value+1] - 1; 
    
    swap (last, position[node]);

    first[node_value+1]--; 
    value[node]++;
}

void bucket_array::decrement(NodeID node) {
    int node_value = value[node];

    swap(first[node_value], position[node]);
    first[node_value]++;
    value[node]--;

    if (value[node] < 0) non_negative++;
}

bool bucket_array::contains(NodeID node) {
    return (value[node] >= 0); 
}

void bucket_array::remove(NodeID node) {
    while (value[node] >= 0) {
        decrement(node);
    }
}

NodeID bucket_array::pickSmallest() {
    if (non_negative >= size) return -1;
    int block = value[array[non_negative]];
    unsigned int pick = random_functions::nextInt(first[block], first[block + 1] - 1);

    return array[pick];
}

void bucket_array::swap(int first_pos, int second_pos) {
    NodeID first = array[first_pos];
    NodeID second = array[second_pos];

    array[first_pos] = second;
    array[second_pos] = first;
    position[first] = second_pos;
    position[second] = first_pos;
}

bool bucket_array::is_empty(int block) {
    return (first[block+1] == first[block]); 
}

