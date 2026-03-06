// NamiLoop 生成的 matmul kernel
// A(64,32) × B(32,64)
// tile(16,4)
for (int i_o = 0; i_o < 4; i_o++) {
    for (int j_o = 0; j_o < 16; j_o++) {
        for (int i_i = 0; i_i < 16; i_i++) {
            for (int j_i = 0; j_i < 4; j_i++) {
                int i = 16 * i_o + i_i;
                int j = 4 * j_o + j_i;
                C[i * 64 + j] = 0;
                for (int k = 0; k < 32; k++) {
                    C[i * 64 + j] += A[i * 32 + k] * B[k * 64 + j];
                }
            }
        }
    }
}
