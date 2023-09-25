#include "global_ocl_env.h"

int main()
{
    init_global_ocl_env("./kernels.cl", {"add", "times"});

    float a = 1, b = 2, c;
    add_buf("a", sizeof(float), 1, &a);
    add_buf("b", sizeof(float), 1, &b);
    add_buf("c", sizeof(float), 1);

    run_kern("add", {1}, "a", "b", "c");
    read_buf(&c, "c");
    printf("%f + %f = %f\n", a, b, c);

    run_kern("times", {1}, a, b, "c");
    read_buf(&c, "c");
    printf("%f * %f = %f\n", a, b, c);

    del_buf("c");

    return 0;
}