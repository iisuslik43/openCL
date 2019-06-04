#define SWAP(a,b) {__local double * tmp=a; a=b; b=tmp;}

__kernel void scan(__global double * input, __global double * output, __global double * sum,
                   __local double * a, __local double * b, int N, int GPU_N)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint block_index = get_group_id(0);
    uint blocks = GPU_N / block_size;

    if (gid < N) {
        a[lid] = b[lid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1)
    {
        if (gid < N) {
            if(lid > (s-1))
            {
                b[lid] = a[lid] + a[lid-s];
            }
            else
            {
                b[lid] = a[lid];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    if (gid < N) {
        output[gid] = a[lid];
    }
    if (lid == block_size - 1) {
        sum[block_index] = a[lid];
    }
}

__kernel void add_sum(__global double * input, __global double * sum, int N, int GPU_N)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
    uint block_index = get_group_id(0);
    uint blocks = GPU_N / block_size;

    if (gid < N && block_index != 0) {
        input[gid] = input[gid] + sum[block_index - 1];
    }
}