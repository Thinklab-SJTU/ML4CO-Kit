#ifdef __cplusplus
extern "C" {
#endif

#define ILS_EXPORT __attribute__((visibility("default")))

ILS_EXPORT void arw(int n, int m, int* xadj, int* adjncy, int* initial_solution, int iterations, int* output);

#ifdef __cplusplus
}
#endif