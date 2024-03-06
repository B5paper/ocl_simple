#include "ocl_simple.h"
#include <iostream>
using namespace std;

int main()
{
    string program_path = "./kernels.cl";
    vector<string> kernel_names{
        "add",
        "times"
    };
    OclEnv ocl_env(program_path, kernel_names);

    float a = 1, b = 2, c;
    ocl_env.add_buf("a", sizeof(float), 1, &a);
    ocl_env.add_buf("b", sizeof(float), 1, &b);
    ocl_env.add_buf("c", sizeof(float), 1);

    ocl_env.run_kernel("add", {1}, "a", "b", "c");
    ocl_env.read_buf(&c, "c");
    printf("%f + %f = %f\n", a, b, c);

    ocl_env.run_kernel("times", {1}, a, b, "c");
    ocl_env.read_buf(&c, "c");
    printf("%f * %f = %f\n", a, b, c);
    
    return 0;
}