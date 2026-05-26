/******************************************************************************
 * mis_config.h
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

#ifndef _STRUCTION_MIS_CONFIG_H_
#define _STRUCTION_MIS_CONFIG_H_

#include <string>
#include <cctype>
#include <algorithm>

#include "definitions.h"

namespace struction {

// Configuration for the calculation of the MIS
struct MISConfig {
    enum Weight_Source {FILE, HYBRID, UNIFORM, GEOMETRIC};
    enum Reduction_Style {NORMAL, DENSE};
    enum Struction_Type {ORIGINAL, MODIFIED, EXTENDED, EXTENDED_REDUCED, NONE};
    enum Key_Type {RANDOM, DEGREE, INCREASE, APPROXIMATE_INCREASE};
    enum Backtrack_Type {IMMEDIATE_TIE_BREAKING, IMMEDIATE_EXCLUDE, END_MIN_KERNEL, NO_BACKTRACK};

    // Name of the graph file.
    std::string graph_filename;
    // Name of the output file.
    std::string output_filename;
    // Name of the kernel file.
    std::string kernel_filename;
    // Seed for the RNG.
    int seed;
    // Time limit for the evolutionary algorithm
    double time_limit;
    // Write the log into a file
    bool print_log;
    // Write the inpendent set into a file
    bool write_graph;
    // Write kernel to file
    bool write_kernel;
    // Write the log into the console
    bool console_log;
    // Number of iterations for the ILS.
    unsigned int ils_iterations;
    // Lower bound for the change rate between best individuals
    double best_limit;
    // Number of candidates for forced insertion.
    unsigned int force_cand;
    // Optimize candidates for ILS.
    bool optimize_candidates;
    // Check graph sortedness
    bool check_sorted;
	// Sort free nodes in local search
	bool sort_freenodes;
	// Perform reduction
	bool perform_reductions;
    // Random weights or read weights from file
    Weight_Source weight_source;
    // Choose reduction order and amount for given graph type
    Reduction_Style reduction_style;


    bool perform_hils;

    Struction_Type struction_type;
    unsigned int struction_degree;
    unsigned int set_limit;
    double global_blow_up_factor;
    double phase_blow_up_factor;
    double key_reinsert_factor;
    unsigned int phase_blow_ups;
    unsigned int max_unimproving_phases;
    Key_Type key_type;
    Backtrack_Type backtrack_style;
    bool reduce_and_peel;
    bool plain_struction;
    bool generalized_fold_disabled;
    bool neighborhood_clique_disabled;
    bool critical_set_disabled;
    bool clique_disabled;
    bool blow_up_disabled;


    void setReductionStyle(const std::string & redu_style) {
        if (strCompare(redu_style, "normal") || strCompare(redu_style, "sparse")) {
            reduction_style = Reduction_Style::NORMAL;
        } else if (strCompare(redu_style, "dense") || strCompare(redu_style, "osm")) {
            reduction_style = Reduction_Style::DENSE;
        }
    }

    void setWeightSource(const std::string & source) {
        if (strCompare(source, "file")) {
            weight_source = Weight_Source::FILE;
        } else if (strCompare(source, "hybrid")) {
            weight_source = Weight_Source::HYBRID;
        } else if (strCompare(source, "uniform")) {
            weight_source = Weight_Source::UNIFORM;
        } else if (strCompare(source, "geometric")) {
            weight_source = Weight_Source::GEOMETRIC;
        }
    }

    void setBacktrackType(const std::string & back_type) {
        if (strCompare(back_type, "immediate_tie_breaking")) {
            backtrack_style = Backtrack_Type ::IMMEDIATE_TIE_BREAKING;
        } else if (strCompare(back_type, "immediate_exclude")) {
            backtrack_style = Backtrack_Type ::IMMEDIATE_EXCLUDE;
        } else if (strCompare(back_type, "end_min_kernel")) {
            backtrack_style = Backtrack_Type::END_MIN_KERNEL;
        } else if (strCompare(back_type, "no_backtrack")) {
            backtrack_style = Backtrack_Type ::NO_BACKTRACK;
        }
    }

    void setStructionType(const std::string & s_type) {
        if (strCompare(s_type, "original")) {
            struction_type = Struction_Type ::ORIGINAL;
        } else if (strCompare(s_type, "modified")) {
            struction_type = Struction_Type ::MODIFIED;
        } else if (strCompare(s_type, "extended")) {
            struction_type = Struction_Type::EXTENDED;
        } else if (strCompare(s_type, "extended_reduced")) {
            struction_type = Struction_Type ::EXTENDED_REDUCED;
        } else if (strCompare(s_type, "none")) {
            struction_type = Struction_Type ::NONE;
        }
    }

    void setKeyType(const std::string & k_type) {
        if (strCompare(k_type, "random")) {
            key_type = Key_Type ::RANDOM;
        } else if (strCompare(k_type, "degree")) {
            key_type = Key_Type ::DEGREE;
        } else if (strCompare(k_type, "increase")) {
            key_type = Key_Type::INCREASE;
        } else if (strCompare(k_type, "approximate_increase")) {
            key_type = Key_Type ::APPROXIMATE_INCREASE;
        }
    }
    private:

    bool strCompare(const std::string & str1, const std::string & str2) {
        return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), [](unsigned char c1, unsigned char c2){ return std::toupper(c1) == std::toupper(c2); });
    }
};

}
#endif
