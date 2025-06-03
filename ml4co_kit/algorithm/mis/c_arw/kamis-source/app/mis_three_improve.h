#ifdef __cplusplus
extern "C" {
#endif

#define ARW_EXPORT __attribute__((visibility("default")))

ARW_EXPORT void mis_three_improve(int n, int* xadj, int* adjncy, int* initial_solution, int* output);

#ifdef __cplusplus
}
#endif