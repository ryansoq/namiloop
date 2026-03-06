// NamiLoop 生成的 matmul kernel
// A(16,8) × B(8,16)
// tile(4,4)
#pragma omp parallel for collapse(2)
for (int i_o = 0; i_o < 4; i_o++) {
    for (int j_o = 0; j_o < 4; j_o++) {
        for (int i_i = 0; i_i < 4; i_i++) {
            for (int j_i = 0; j_i < 4; j_i++) {
                int i = 4 * i_o + i_i;
                int j = 4 * j_o + j_i;
                C[i * 16 + j] = 0;
                for (int k = 0; k < 8; k++) {
                    C[i * 16 + j] += A[i * 8 + k] * B[k * 16 + j];
                }
            }
        }
    }
}
