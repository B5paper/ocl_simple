#include "ocl_simple.h"
#include <iostream>
using namespace std;

int main()
{
    OclEnv ocl_env;
    string program_path = "./kernels.cl";
    vector<string> kernel_names{
        "add"
    };
    init_ocl(ocl_env, program_path, kernel_names);
    float a = 1, b = 2, c;
    env_add_buf(ocl_env, "a", sizeof(float), 1);
    env_add_buf(ocl_env, "b", sizeof(float), 1);
    env_add_buf(ocl_env, "c", sizeof(float), 1);
    write_buf("a", &a, ocl_env);
    write_buf("b", &b, ocl_env);
    run_kernel("add", ocl_env, {1}, "a", "b", "c");
    read_buf(&c, "c", ocl_env);
    printf("%f + %f = %f\n", a, b, c);
    return 0;
}