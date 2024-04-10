#include "simple_ocl.hpp"

int main()
{
    init_ocl_env("kernels.cl", {"vec_add"});
    float *A = (float*) add_buf_mem("A", sizeof(float), 4);
    for (int i = 0; i < 4; ++i)
        *(A+i) = i;
    float *B = (float*) add_buf_mem("B", sizeof(float), 4, A);
    float *C = (float*) add_buf_mem("C", sizeof(float), 4);
    sync_cpu_to_gpu({"A", "B"});
    run_kern("vec_add", {4}, "A", "B", "C");
    sync_gpu_to_cpu({"A", "B", "C"});
    printf("A: ");
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", *(A+i));
    putchar('\n');
    printf("B: ");
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", *(B+i));
    putchar('\n');
    printf("C: ");
    for (int i = 0; i < 4; ++i)
        printf("%.2f, ", *(C+i));
    putchar('\n');
    return 0;
}