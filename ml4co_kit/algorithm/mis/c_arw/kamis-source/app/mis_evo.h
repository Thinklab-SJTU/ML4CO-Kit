#ifdef __cplusplus
extern "C" {
#endif

#define EVO_EXPORT __attribute__((visibility("default")))

EVO_EXPORT void mis_evo(int n, int* xadj, int* adjncy, double time_limit, int* output);

#ifdef __cplusplus
}
#endif