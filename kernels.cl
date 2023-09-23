kernel void add(global float *A, global float *B, global float *C)
{
    size_t dim_0 = get_global_id(0);
    C[dim_0] = A[dim_0] + B[dim_0];
}