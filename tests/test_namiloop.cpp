// ============================================================================
// NamiLoop 測試
// ============================================================================
#include "../include/namiloop/namiloop.hpp"
#include <cassert>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace namiloop;

int tests_passed = 0;
int tests_total = 0;

#define TEST(name) \
    void name(); \
    struct name##_reg { name##_reg() { tests_total++; } } name##_inst; \
    void name()

#define RUN_TEST(name) \
    do { \
        std::cout << "  [" << #name << "] "; \
        name(); \
        tests_passed++; \
        std::cout << "✅ PASS\n"; \
    } while(0)

#define ASSERT(cond) \
    do { if (!(cond)) { std::cout << "❌ FAIL: " #cond "\n"; return; } } while(0)

// --- 測試定義 ---

TEST(test_tensor_create) {
    Tensor A("A", 100, 50);
    ASSERT(A.name() == "A");
    ASSERT(A.rows() == 100);
    ASSERT(A.cols() == 50);
}

TEST(test_expr_matmul) {
    Tensor A("A", 100, 50);
    Tensor B("B", 50, 100);
    auto C = A * B;
    ASSERT(C.op_ == Expr::MATMUL);
    ASSERT(C.lhs_->name() == "A");
    ASSERT(C.rhs_->name() == "B");
    ASSERT(C.loops_.size() == 3);  // i, j, k
}

TEST(test_codegen_basic) {
    Tensor A("A", 4, 2);
    Tensor B("B", 2, 4);
    auto C = A * B;
    std::string code = C.codegen();
    // 應包含基本迴圈結構
    ASSERT(code.find("for (int i") != std::string::npos);
    ASSERT(code.find("for (int j") != std::string::npos);
    ASSERT(code.find("for (int k") != std::string::npos);
    ASSERT(code.find("C[") != std::string::npos);
    ASSERT(code.find("A[") != std::string::npos);
    ASSERT(code.find("B[") != std::string::npos);
}

TEST(test_split) {
    Tensor A("A", 8, 4);
    Tensor B("B", 4, 8);
    auto C = A * B;
    C.split(0, 4);  // split i by 4
    std::string code = C.codegen();
    // split i → 應有 i_o, i_i
    ASSERT(code.find("i_o") != std::string::npos);
    ASSERT(code.find("i_i") != std::string::npos);
}

TEST(test_tile) {
    Tensor A("A", 8, 4);
    Tensor B("B", 4, 8);
    auto C = A * B;
    C.tile(4, 4);
    std::string code = C.codegen();
    // tile → 應有外層 i_o, j_o 和內層 i_i, j_i
    ASSERT(code.find("i_o") != std::string::npos);
    ASSERT(code.find("j_o") != std::string::npos);
    ASSERT(code.find("i_i") != std::string::npos);
    ASSERT(code.find("j_i") != std::string::npos);
    // 仿射索引（ax + b）
    ASSERT(code.find("4 * i_o + i_i") != std::string::npos);
    ASSERT(code.find("4 * j_o + j_i") != std::string::npos);
}

TEST(test_parallel) {
    Tensor A("A", 8, 4);
    Tensor B("B", 4, 8);
    auto C = A * B;
    C.parallel();
    std::string code = C.codegen();
    ASSERT(code.find("#pragma omp parallel for") != std::string::npos);
}

TEST(test_kernel_correctness) {
    // 用小矩陣驗證 codegen 的 kernel 計算正確
    const int M = 4, K = 3, N = 4;
    
    double A[M * K] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    double B[K * N] = {1,0,0,1, 0,1,0,0, 0,0,1,0};
    double C[M * N] = {0};
    double C_ref[M * N] = {0};
    
    // 參考結果（直接計算）
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            C_ref[i * N + j] = 0;
            for (int k = 0; k < K; k++)
                C_ref[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    
    // 用 codegen 的 kernel（寫入暫存檔、編譯、執行）
    Tensor tA("A", M, K);
    Tensor tB("B", K, N);
    auto expr = tA * tB;
    expr.tile(2, 2);
    std::string kernel = expr.codegen();
    
    // 生成驗證程式
    std::ostringstream prog;
    prog << "#include <iostream>\n#include <cmath>\n\n";
    prog << "int main() {\n";
    prog << "    const int M=" << M << ", K=" << K << ", N=" << N << ";\n";
    prog << "    double A[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};\n";
    prog << "    double B[] = {1,0,0,1, 0,1,0,0, 0,0,1,0};\n";
    prog << "    double C[" << M*N << "] = {0};\n";
    prog << "    double C_ref[] = {";
    for (int i = 0; i < M*N; i++) {
        if (i > 0) prog << ",";
        prog << C_ref[i];
    }
    prog << "};\n\n";
    prog << kernel << "\n";
    prog << "    for (int i = 0; i < " << M*N << "; i++) {\n";
    prog << "        if (std::abs(C[i] - C_ref[i]) > 1e-9) {\n";
    prog << "            std::cout << 0 << std::endl;\n";
    prog << "            return 1;\n";
    prog << "        }\n";
    prog << "    }\n";
    prog << "    std::cout << 1 << std::endl;\n";
    prog << "    return 0;\n";
    prog << "}\n";
    
    // 編譯執行
    {
        std::ofstream f("/tmp/namiloop_test_correctness.cpp");
        f << prog.str();
    }
    int ret = std::system("g++ -std=c++17 -O2 -o /tmp/namiloop_test_correctness /tmp/namiloop_test_correctness.cpp 2>/dev/null");
    ASSERT(ret == 0);
    
    FILE* pipe = popen("/tmp/namiloop_test_correctness", "r");
    ASSERT(pipe != nullptr);
    char buf[64];
    ASSERT(fgets(buf, sizeof(buf), pipe) != nullptr);
    pclose(pipe);
    ASSERT(std::atoi(buf) == 1);
    
    // 清理
    std::remove("/tmp/namiloop_test_correctness.cpp");
    std::remove("/tmp/namiloop_test_correctness");
}

int main() {
    std::cout << "\n🧪 NamiLoop 測試\n";
    std::cout << "==================\n";
    
    RUN_TEST(test_tensor_create);
    RUN_TEST(test_expr_matmul);
    RUN_TEST(test_codegen_basic);
    RUN_TEST(test_split);
    RUN_TEST(test_tile);
    RUN_TEST(test_parallel);
    RUN_TEST(test_kernel_correctness);
    
    std::cout << "\n==================\n";
    std::cout << "✅ " << tests_passed << "/7 tests passed\n\n";
    
    return (tests_passed == 7) ? 0 : 1;
}
