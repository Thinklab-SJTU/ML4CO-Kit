#ifdef __cplusplus
extern "C" {
#endif

#define ILS_EXPORT __attribute__((visibility("default")))

ILS_EXPORT void mis_ils(
    int n, int* xadj, int* adjncy, int* initial_solution, 
    int iterations, int use_three_ls, int* output
);

#ifdef __cplusplus
}
#endif