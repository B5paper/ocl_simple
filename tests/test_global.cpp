#include "global_ocl_env.h"

#include <vector>
using namespace std;
void test()
{
    const int N = 99999999;
    vector<float> a(N), b(N), c(N);
    for (int i = 0; i < N; ++i)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
    add_buf("a", sizeof(float), N, a.data());
    add_buf("b", sizeof(float), N, b.data());
    add_buf("c", sizeof(float), N);
    run_kern("add", {N}, "a", "b", "c");
    read_buf(c.data(), "c");
    for (int i = N - 1; i > N - 6; --i)
    {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }
}

int main()
{
    init_global_ocl_env("./kernels.cl", {"add", "times"});

    float A[512], B[512], C[512];
    for (int i = 0; i < 512; ++i)
    {
        A[i] = (float)rand() / RAND_MAX * 100;
        B[i] = (float)rand() / RAND_MAX * 100;
    }
    add_buf("A", sizeof(float), 512, A);
    add_buf("B", sizeof(float), 512, B);
    add_buf("C", sizeof(float), 512);
    run_kern("add", {512}, "A", "B", "C");
    read_buf(C, "C");

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