/******************************************************************************
 * mis_log.h
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

#ifndef _STRUCTION_MIS_LOG_H_
#define _STRUCTION_MIS_LOG_H_

#include <sstream>

#include "timer.h"
#include "mmwis_config.h"
#include "graph_access.h"

class mis_log {
    public:
        /**
         * Get the singleton logger instance.
         * 
         * @return Instance of the logger.
         */
        static mis_log *instance() {
            static mis_log inst;
            return &inst;
        };

        /**
         * Set the config.
         *
         * @param config Config for the evolutionary algorithm.
         */
        void set_config(::mmwis::MISConfig & config);

        /**
         * Set the graph.
         *
         * @param G Graph representation.
         */
        template <typename graph>
        void set_graph(graph & G);

        /**
         * Write the log to a file.
         */
        void write_log();

        /**
         * Add a newline to the log.
         */
        void print_newline();

        /**
         * Print information about the graph.
         */
        void print_graph();

        /**
         * Print the current config.
         */
        void print_config();

        /**
         * Print information about the reduction step.
         * Includes number of extracted nodes and resulting kernel size
         *
         * @param mis_config Config for the logger.
         * @param extracted_nodes Number of removed nodes.
         * @param kernel_size Number of remaining nodes.
         */
        void print_reduction(::mmwis::MISConfig & mis_config, unsigned int extracted_nodes, unsigned int kernel_size);

        /**
         * Print banner for the initialization phase.
         */
        void print_init_title();

        /**
         * Print the final results.
         */
        void print_results();

        /**
         * Print a title.
         */
        void print_title();
        
        /**
         * Restart the timer for the total time including IO, etc.
         */
        void restart_total_timer();

        /**
         * Update the size of the best solution.
         *
         * @param mis_config Config for the logger.
         * @param size Candidate to replace the best solution size.
         */
        void set_best_size(::mmwis::MISConfig & mis_config, unsigned int size);

        /**
         * Reset the size of the best solution.
         */
        void reset_best_size();

    private:
        // General information
        timer total_timer;
        std::stringstream filebuffer_string;
        ::mmwis::MISConfig log_config;

        // Graph informations
        std::string graph_name;
        unsigned int number_of_nodes;
        unsigned int number_of_edges;
        unsigned int arc_scans;
        double avg_degree;
        double density;

        // Reduction information
        unsigned int current_is_size;
        unsigned int optimum_size;

        // Results information
        double total_time_taken;
        double time_taken_best;

        /**
         * Default Constructor.
         */
        mis_log();

        /**
         * Default Destructor.
         */
        virtual ~mis_log();
};


template <typename graph>
inline void mis_log::set_graph(graph& G) {
    number_of_nodes = G.number_of_nodes();
    number_of_edges = G.number_of_edges();
    avg_degree = (double) number_of_edges / number_of_nodes;
    density = (double) (1 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1));
}


#endif
