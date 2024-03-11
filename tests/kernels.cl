kernel void add(global float *A, global float *B, global float *C)
{
    size_t dim_0 = get_global_id(0);
    C[dim_0] = A[dim_0] + B[dim_0];
}

kernel void times(float a, float b, global float *c)
{
    *c = a * b;
}

kernel void assign_A_to_B(global int *A, global int *B)
{
	*B = *A;
}