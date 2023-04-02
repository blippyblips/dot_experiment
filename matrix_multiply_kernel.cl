__kernel void matrix_multiply(__global const float* A,
                              __global const float* B,
                              __global float* C,
                              int m) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;

    for (int k = 0; k < m; ++k) {
        sum += A[i * m + k] * B[k * get_global_size(1) + j];
    }

    C[i * get_global_size(1) + j] = sum;
}
