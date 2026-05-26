/******************************************************************************
 * fast_set.h
 *
 * Copyright (C) 2015-2017 Darren Strash <strash@kit.edu>
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

#ifndef FAST_SET_H
#define FAST_SET_H

#include <vector>
#include <cstddef>

class fast_set {

public:
    std::vector<int> used;
    int uid;

    fast_set(int const n) : used(n, 0), uid(1) { }

    void clear() {
        uid++;
        if (uid < 0) {
            for (unsigned int i = 0; i < used.size(); i++) used[i] = 0;
            uid = 1;
        }
    }

    bool add(int i) {
        bool const res(used[i] != uid);
        used[i] = uid;
        return res;
    }

    void remove(int i) {
        used[i] = uid - 1;
    }

    bool get(int i) const {
        return (used[i] == uid);
    }

    void resize(size_t size) {
        used.resize(size, uid - 1);
    }

    size_t size() {
        return used.size();
    }
};

#endif // FAST_SET_H
