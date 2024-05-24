kernel void vec_add(global float *A, global float *B, global float *output)
{
    size_t gid = get_global_id(0);
    output[gid] = A[gid] + B[gid];
}

kernel void mat_mul(global float *M_A, global float *M_B, global float *output,
    const uint n_row_A, const uint n_col_A, const uint n_col_B)
{
    size_t gid = get_global_id(0);
    for (uint p = 0; p < n_col_B; ++p)
    {
        float prod_sum = 0.0f;
        for (uint k = 0; k < n_col_A; ++k)
        {
            prod_sum += M_A[gid * n_col_A + k] * M_B[k * n_col_B + p];
        }
        output[gid * n_col_B + p] = prod_sum;
    }
}

kernel void mat_mul_1d_span(global float *M_A, global float *M_B, global float *output,
    const uint n_row_A, const uint n_col_A, const uint n_col_B)
{
    size_t glb_id = get_global_id(0);
    size_t glb_num = get_global_size(0);
    size_t glb_span_size = n_row_A / glb_num;
    size_t start = glb_span_size * glb_id,
        end = glb_span_size * (glb_id + 1);
    for (int row_A = start; row_A < end; ++row_A)
    {
        for (int col_B = 0; col_B < n_col_B; ++col_B)
        {
            float prod_sum = 0.0f;
            for (int k = 0; k < n_col_A; ++k)
            {
                prod_sum += M_A[row_A * n_col_A + k] * M_B[k * n_col_B + col_B];
            }
            output[row_A * n_col_A + col_B] = prod_sum;;
        }
    }
}

kernel void mat_mul_2d(global float *M_A, global float *M_B, global float *output,
    const uint n_row_A, const uint n_col_A, const uint n_col_B)
{
    size_t gid_0 = get_global_id(0);
    size_t gid_1 = get_global_id(1);
    float prod_sum = 0.0f;
    for (uint k = 0; k < n_col_A; ++k)
    {
        prod_sum += M_A[gid_0 * n_col_A + k] * M_B[k * n_col_B + gid_1];
    }
    output[gid_0 * n_col_B + gid_1] = prod_sum;
}

void vec_dot(local float *vec_1, local float *vec_2, local float *output, const uint vec_len)
{
    *output = 0;
    for (uint i = 0; i < vec_len; ++i)
    {
        *output += vec_1[i] * vec_2[i];
    }
}

kernel void mat_mul_ocl_mem_opt(global float *M_A, global float *M_B, global float *output,
    const uint n_row_A, const uint n_col_A, const uint n_col_B,
    local float *vec_B)
{
    size_t glb_id = get_global_id(0);
    size_t grp_id = get_group_id(0);
    size_t loc_id = get_local_id(0);
    size_t grp_size = get_local_size(0);
    size_t num_grps = n_row_A / grp_size;
    size_t micro_len = n_col_A / grp_size;

    // copy from global to private
    private float vec_A[1024];
    for (int i = 0; i < n_col_A; ++i)
        vec_A[i] = M_A[glb_id * n_col_A + i];  // M_A 的第 glb_id 行，第 i 列
    
    for (int i = 0; i < n_col_B; ++i)
    {
        // copy from global to local shared
        for (int j = 0; j < micro_len; ++j)
        {
            vec_B[loc_id * micro_len + j] = M_B[(loc_id * micro_len + j) * n_col_B + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // calc vector dot product, and store the result into global memory
        float prod_sum = 0.0f;
        for (int k = 0; k < n_col_A; ++k)
            prod_sum += vec_A[k] * vec_B[k];
        output[glb_id * n_col_B + i] = prod_sum;   
    }
}
