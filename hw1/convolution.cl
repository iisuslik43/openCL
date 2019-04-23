__kernel void convolution(__global double * a, __global double * b, __global double * c, int n, int m) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= n || j >= n)
        return;

    double sum = 0;
    int hm = (m - 1) / 2;
    for (int k = -hm; k <= hm; k++) {
        for (int l = -hm; l <= hm; l++) {
            if (i + k >= 0 && i + k < n && j + l >= 0 && j + l < n) {
                sum += a[(i + k) * n + j + l] * b[(k + hm) * m + l + hm];
            }
        }
    }

    c[i * n + j] = sum;
}