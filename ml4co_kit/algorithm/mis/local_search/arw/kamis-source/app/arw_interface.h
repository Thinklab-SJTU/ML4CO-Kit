#ifdef __cplusplus
extern "C" {
#endif

#define ARW_EXPORT __attribute__((visibility("default")))

ARW_EXPORT void arw_1iter(int n, int m, int* xadj, int* adjncy, int* initial_solution, int* output);

#ifdef __cplusplus
}
#endif