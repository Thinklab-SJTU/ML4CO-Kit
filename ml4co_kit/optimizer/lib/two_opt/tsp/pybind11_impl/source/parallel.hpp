#pragma once

#include <cstdint>
#include <thread>

// Type aliases for integer types
using int64 = int64_t;
using int32 = int32_t;

/**
 * @brief Parallel execution helper template struct
 * 
 * @tparam CallableType Type of the callable function/lambda
 * @tparam IntableType Type for range/index (must be integer-like)
 * @tparam _workers Default number of worker threads
 * @tparam _enable_fn_copy Whether to copy the function object (true) or use reference (false)
 */
template<typename CallableType, typename IntableType, int32 _workers, 
         bool _enable_fn_copy>
struct Parallelizer {
    /**
     * @brief Execute function in parallel with function copy
     * 
     * Creates worker threads and distributes the range [0, range) among them.
     * Each thread executes fn(i) for its assigned indices.
     * 
     * @param fn Callable function to execute for each index
     * @param range Total number of tasks (indices from 0 to range-1)
     * @param workers Number of worker threads to use
     */
    inline __attribute__((always_inline)) 
    static void sync_perform_with_fn_copy(const CallableType fn, 
                                          const IntableType range, 
                                          const int32 workers = _workers) {
        std::thread threads[workers];
        const IntableType task_num_per_worker = range / workers;
        const IntableType remainder = range % workers;
        IntableType next_task = 0;
        
        // Launch worker threads
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id] = std::thread([id, &fn, next_task, &task_num_per_worker, 
                                       &remainder, &range]() {
                // Calculate the end index for this worker
                const IntableType _end = next_task + task_num_per_worker + (id < remainder);
                
                // Execute function for each assigned index
                #pragma unroll 8
                for (IntableType i = next_task; i < _end; ++i) {
                    fn(i);
                }
            });
            // Update next_task for the next worker
            next_task += task_num_per_worker + (id < remainder);
        }
        
        // Wait for all threads to complete
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id].join();
        }
    }

    /**
     * @brief Execute function in parallel without function copy (uses reference)
     * 
     * Similar to sync_perform_with_fn_copy, but uses a reference to the function
     * instead of copying it. This is more efficient for large callable objects.
     * 
     * @param fn Reference to callable function to execute for each index
     * @param range Total number of tasks (indices from 0 to range-1)
     * @param workers Number of worker threads to use
     */
    inline __attribute__((always_inline)) 
    static void sync_perform_without_fn_copy(const CallableType & fn, 
                                             const IntableType range, 
                                             const int32 workers = _workers) {
        std::thread threads[workers];
        const IntableType task_num_per_worker = range / workers;
        const IntableType remainder = range % workers;
        IntableType next_task = 0;
        
        // Launch worker threads
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id] = std::thread([id, &fn, next_task, &task_num_per_worker, 
                                       &remainder]() {
                // Calculate the end index for this worker
                const IntableType _end = next_task + task_num_per_worker + (id < remainder);
                
                // Execute function for each assigned index
                #pragma unroll 8
                for (IntableType i = next_task; i < _end; ++i) {
                    fn(i);
                }
            });
            // Update next_task for the next worker
            next_task += task_num_per_worker + (id < remainder);
        }
        
        // Wait for all threads to complete
        #pragma unroll 8
        for (IntableType id = 0; id < workers; ++id) {
            threads[id].join();
        }
    }

    /**
     * @brief Operator() to execute parallel work
     * 
     * Selects between copy and reference version based on _enable_fn_copy template parameter.
     * Uses C++23 syntax if available, otherwise falls back to non-static method.
     * 
     * @param fn Callable function to execute
     * @param range Total number of tasks
     * @param workers Number of worker threads
     */
    // If C++ standard >= 23, we can define a static operator()
    #if __cplusplus >= 202302L
    static inline __attribute__((always_inline)) 
    void operator()(const CallableType & fn, 
                    const IntableType range, 
                    const int32 workers = _workers) {
        if constexpr (_enable_fn_copy) {
            sync_perform_with_fn_copy(fn, range, workers);
        } else {
            sync_perform_without_fn_copy(fn, range, workers);
        }
    }
    #else
    inline __attribute__((always_inline)) 
    void operator()(const CallableType & fn, 
                    const IntableType range, 
                    const int32 workers = _workers) {
        if constexpr (_enable_fn_copy) {
            sync_perform_with_fn_copy(fn, range, workers);
        } else {
            sync_perform_without_fn_copy(fn, range, workers);
        }
    }
    #endif
};

/**
 * @brief Convenience function for parallel execution
 * 
 * Executes a callable function for indices in range [0, range) in parallel.
 * Falls back to sequential execution if workers <= 1 or _turnoff is true.
 * 
 * @tparam CallableType Type of the callable function/lambda
 * @tparam IntableType Type for range/index (must be integer-like)
 * @tparam _workers Default number of worker threads (default: 20)
 * @tparam _enable_fn_copy Whether to copy the function object (default: false)
 * @tparam _turnoff Debug flag to disable parallelization (default: false)
 * 
 * @param fn Callable function to execute for each index
 * @param range Total number of tasks (indices from 0 to range-1)
 * @param workers Number of worker threads to use
 */
template<typename CallableType, typename IntableType, 
         int32 _workers = 20, bool _enable_fn_copy = false, bool _turnoff = false>
inline __attribute__((always_inline)) 
void parallelize(const CallableType & fn, 
                 const IntableType range, 
                 const int32 workers = _workers) {
    if constexpr (!_turnoff) {
        // Use sequential execution if only one worker
        if (workers <= 1) {
            for (IntableType i = 0; i < range; ++i) {
                fn(i);
            }
        } else {
            // Use parallel execution with multiple workers
            Parallelizer<CallableType, IntableType, _workers, _enable_fn_copy>{}(fn, range, workers);
        }
    } else {
        // Debug mode: force sequential execution
        for (IntableType i = 0; i < range; ++i) {
            fn(i);
        }
    }
}
