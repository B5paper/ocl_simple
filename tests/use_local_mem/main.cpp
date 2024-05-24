#include "../../simple_ocl.hpp"
#include <string.h>
#include <time.h>
#include <stdio.h>

void print_mat(float *M, int n_row, int n_col)
{
    for (int i = 0; i < n_row; ++i)
    {
        for (int j = 0; j < n_col; ++j)
        {
            if (j != n_col - 1)
                printf("%.1f, ", M[i * n_col + j]);
            else
                printf("%.1f", M[i * n_col + j]);
        }
        putchar('\n');
    }
}

float timeit_ms(const char *cmd)
{
    static timespec tmspec_start, tmspec_end;
    if (strcmp(cmd, "start") == 0)
    {
        timespec_get(&tmspec_start, TIME_UTC);
        return 0.0f;
    }
    else if (strcmp(cmd, "end") == 0)
    {
        timespec_get(&tmspec_end, TIME_UTC);
        // float start_ms = (float) tmspec_start.tv_sec * 1000.0f + (float) tmspec_start.tv_nsec / 1000.0f / 1000.0f;
        // float end_ms = (float) tmspec_end.tv_sec * 1000.0f + (float) tmspec_end.tv_nsec / 1000.0f / 1000.0f;
        float dur_ms = (float) (tmspec_end.tv_sec - tmspec_start.tv_sec) * 1000.0f 
            + (float) (tmspec_end.tv_nsec - tmspec_start.tv_nsec) / 1000.0f / 1000.0f;
        return dur_ms;
    }
    else
    {
        printf("unknown timeit_ms() command\n");
        return -1.0;
    }
}

void test_mat_mul_cpu(float *A, float *B, float C[],
    int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
{
    timeit_ms("start");
    for (int i = 0; i < n_row_A; ++i)
    {
        for (int j = 0; j < n_col_B; ++j)
        {
            float prod = 0.0f;
            for (int k = 0; k < n_col_A; ++k)
            {
                prod += A[i * n_col_A + k] * B[k * n_col_B + j];
            }
            C[i * n_col_B + j] = prod;
        }
    }
    float dur_ms = timeit_ms("end");

    if (disp_mat)
    {
        printf("A:\n");
        print_mat(A, n_row_A, n_col_A);
        putchar('\n');
        printf("B:\n");
        print_mat(B, n_col_A, n_col_B);
        putchar('\n');
        printf("C:\n");
        print_mat(C, n_row_A, n_col_B);
        putchar('\n');
    }

    printf("time duration: %.3f ms\n", dur_ms);
}

void test_mat_mul_ocl_mem_opt(float A[], float B[], float C[],
    int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
{
    init_ocl_env("./kernels.cl", {"mat_mul_ocl_mem_opt"});
    int elm_num_A = n_row_A * n_col_A;
    int elm_num_B = n_col_A * n_col_B;
    int elm_num_C = n_row_A * n_col_B;
    add_buf("A", sizeof(float), elm_num_A);
    add_buf("B", sizeof(float), elm_num_B);
    add_buf("C", sizeof(float), elm_num_C);
    add_local_buf("vec_B", sizeof(float), 1024);

    timeit_ms("start");
    write_buf("A", A);
    write_buf("B", B);
    run_kern("mat_mul_ocl_mem_opt",
        {(size_t) n_row_A}, {(size_t) n_row_A / 8},
        "A", "B", "C", n_row_A, n_col_A, n_col_B,
        "vec_B");
    read_buf(C, "C");
    float dur_ms = timeit_ms("end");

    if (disp_mat)
    {
        printf("A:\n");
        print_mat(A, n_row_A, n_col_A);
        putchar('\n');
        printf("B:\n");
        print_mat(B, n_col_A, n_col_B);
        putchar('\n');
        printf("C:\n");
        print_mat(C, n_row_A, n_col_B);
        putchar('\n');
    }

    printf("time duration: %.3f ms\n", dur_ms);
    exit_ocl_env();
}

void gen_mat(vector<float> &A, vector<float> &B, vector<float> &C,
    int n_row_A, int n_col_A, int n_col_B)
{
    int elm_num_A = n_row_A * n_col_A;
    int elm_num_B = n_col_A * n_col_B;
    int elm_num_C = n_row_A * n_col_B;
    A.resize(elm_num_A);
    B.resize(elm_num_B);
    C.resize(elm_num_C);
    for (int i = 0; i < elm_num_A; ++i)
        A[i] = rand() % 10;
    for (int i = 0; i < elm_num_B; ++i)
        B[i] = rand() % 10;
}

void test_mat_mul_ocl(float A[], float B[], float C[],
    int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
{
    init_ocl_env("./kernels.cl", {"mat_mul"});
    int elm_num_A = n_row_A * n_col_A;
    int elm_num_B = n_col_A * n_col_B;
    int elm_num_C = n_row_A * n_col_B;
    add_buf("A", sizeof(float), elm_num_A);
    add_buf("B", sizeof(float), elm_num_B);
    add_buf("C", sizeof(float), elm_num_C);
    timeit_ms("start");
    write_buf("A", A);
    write_buf("B", B);
    run_kern("mat_mul", {(size_t) n_row_A}, "A", "B", "C",
        n_row_A, n_col_A, n_col_B);
    read_buf(C, "C");
    float dur_ms = timeit_ms("end");

    if (disp_mat)
    {
        printf("A:\n");
        print_mat(A, n_row_A, n_col_A);
        putchar('\n');
        printf("B:\n");
        print_mat(B, n_col_A, n_col_B);
        putchar('\n');
        printf("C:\n");
        print_mat(C, n_row_A, n_col_B);
        putchar('\n');
    }
    printf("time duration: %.3f ms\n", dur_ms);
    exit_ocl_env();
}

void test_mat_mul_1d_span(float A[], float B[], float C[],
    int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
{
    init_ocl_env("./kernels.cl", {"mat_mul_1d_span"});
    int elm_num_A = n_row_A * n_col_A;
    int elm_num_B = n_col_A * n_col_B;
    int elm_num_C = n_row_A * n_col_B;
    add_buf("A", sizeof(float), elm_num_A);
    add_buf("B", sizeof(float), elm_num_B);
    add_buf("C", sizeof(float), elm_num_C);
    timeit_ms("start");
    write_buf("A", A);
    write_buf("B", B);
    run_kern("mat_mul_1d_span", {(size_t) 512}, "A", "B", "C",
        n_row_A, n_col_A, n_col_B);
    read_buf(C, "C");
    float dur_ms = timeit_ms("end");

    if (disp_mat)
    {
        printf("A:\n");
        print_mat(A, n_row_A, n_col_A);
        putchar('\n');
        printf("B:\n");
        print_mat(B, n_col_A, n_col_B);
        putchar('\n');
        printf("C:\n");
        print_mat(C, n_row_A, n_col_B);
        putchar('\n');
    }
    printf("time duration: %.3f ms\n", dur_ms);
    exit_ocl_env();
}

void compare_vec(const vector<float> &candi, const vector<float> &ref)
{
    bool bad_result = false;
    for (int i = 0; i < ref.size(); ++i)
    {
        if (ref[i] != candi[i])
        {
            printf("not equal, %d idx, candi_val: %f vs ref_val: %f\n", i, candi[i], ref[i]);
            bad_result = true;
        }
    }
    if (!bad_result)
        printf("[checked] gpu calc results are all correct.\n");
}

void test_mat_mul_2d(float A[], float B[], float C[],
    int n_row_A, int n_col_A, int n_col_B, bool disp_mat)
{
    init_ocl_env("./kernels.cl", {"mat_mul_2d"});
    int elm_num_A = n_row_A * n_col_A;
    int elm_num_B = n_col_A * n_col_B;
    int elm_num_C = n_row_A * n_col_B;
    add_buf("A", sizeof(float), elm_num_A);
    add_buf("B", sizeof(float), elm_num_B);
    add_buf("C", sizeof(float), elm_num_C);
    timeit_ms("start");
    write_buf("A", A);
    write_buf("B", B);
    run_kern("mat_mul_2d", {(size_t) 1024, (size_t) 1024},
        "A", "B", "C", n_row_A, n_col_A, n_col_B);
    read_buf(C, "C");
    float dur_ms = timeit_ms("end");

    if (disp_mat)
    {
        printf("A:\n");
        print_mat(A, n_row_A, n_col_A);
        putchar('\n');
        printf("B:\n");
        print_mat(B, n_col_A, n_col_B);
        putchar('\n');
        printf("C:\n");
        print_mat(C, n_row_A, n_col_B);
        putchar('\n');
    }
    printf("time duration: %.3f ms\n", dur_ms);
    exit_ocl_env();
}

int main()
{
    vector<float> A, B, C;
    int n_row_A = 1024, n_col_A = 1024, n_col_B = 1024;
    gen_mat(A, B, C, n_row_A, n_col_A, n_col_B);
    printf("matrices generated. A[%d][%d], B[%d][%d]\n", n_row_A, n_col_A, n_col_A, n_col_B);
    putchar('\n');

    printf("-------- test_mat_mul_cpu --------\n");
    test_mat_mul_cpu(A.data(), B.data(), C.data(), n_row_A, n_col_A, n_col_B, false);
    putchar('\n');
    vector<float> C_ref(C);

    printf("-------- test_mat_mul_ocl --------\n");
    test_mat_mul_ocl(A.data(), B.data(), C.data(), n_row_A, n_col_A, n_col_B, false);
    compare_vec(C, C_ref);
    putchar('\n');

    printf("-------- test_mat_mul_1d_span --------\n");
    test_mat_mul_1d_span(A.data(), B.data(), C.data(), n_row_A, n_col_A, n_col_B, false);
    compare_vec(C, C_ref);
    putchar('\n');

    printf("-------- test_mat_mul_2d --------\n");
    test_mat_mul_2d(A.data(), B.data(), C.data(), n_row_A, n_col_A, n_col_B, false);
    compare_vec(C, C_ref);
    putchar('\n');
    
    printf("-------- test_mat_mul_2d_mem_opt --------\n");
    test_mat_mul_ocl_mem_opt(A.data(), B.data(), C.data(), n_row_A, n_col_A, n_col_B, false);
    compare_vec(C, C_ref);
    putchar('\n');

    return 0;
}