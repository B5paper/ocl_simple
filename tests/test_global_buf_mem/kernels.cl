kernel void vec_add(global float *A, global float *B, global float *C)
{
    size_t id = get_global_id(0);
    C[id] = A[id] + B[id];
}