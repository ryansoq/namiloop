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

#define RUN_TEST(name) \
    do { \
        std::cout << "  [" << #name << "] "; \
        name(); \
        tests_passed++; \
        std::cout << "✅ PASS\n"; \
    } while(0)

#define ASSERT(cond) \
    do { if (!(cond)) { std::cerr << "❌ FAIL: " #cond " (" __FILE__ ":" << __LINE__ << ")\n"; exit(1); } } while(0)

// --- test_tensor_create ---
void test_tensor_create() {
    Tensor A("A", 100, 50);
    ASSERT(A.name() == "A");
    ASSERT(A.rows() == 100);
    ASSERT(A.cols() == 50);
}

// --- test_expr_matmul ---
void test_expr_matmul() {
    Tensor A("A", 100, 50);
    Tensor B("B", 50, 100);
    auto C = A * B;
    ASSERT(C.op_ == Expr::MATMUL);
    ASSERT(C.lhs_->name() == "A");
    ASSERT(C.rhs_->name() == "B");
    // 初始 loop order: i, j, k
    ASSERT(C.loop_order_.size() == 3);
    ASSERT(C.loop_order_[0]->name() == "i");
    ASSERT(C.loop_order_[1]->name() == "j");
    ASSERT(C.loop_order_[2]->name() == "k");
}

// --- test_loops_api ---
void test_loops_api() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();
    ASSERT(i->name() == "i");
    ASSERT(i->extent() == 16);
    ASSERT(j->name() == "j");
    ASSERT(j->extent() == 16);
    ASSERT(k->name() == "k");
    ASSERT(k->extent() == 8);
}

// --- test_split ---
void test_split() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();

    auto [i_o, i_i] = i->split(4);
    ASSERT(i_o->name() == "i_o");
    ASSERT(i_o->extent() == 4);   // 16/4
    ASSERT(i_i->name() == "i_i");
    ASSERT(i_i->extent() == 4);

    // loop order 應該是: i_o, i_i, j, k
    ASSERT(C.loop_order_.size() == 4);
    ASSERT(C.loop_order_[0]->name() == "i_o");
    ASSERT(C.loop_order_[1]->name() == "i_i");
    ASSERT(C.loop_order_[2]->name() == "j");
    ASSERT(C.loop_order_[3]->name() == "k");

    // codegen 應含仿射索引
    std::string code = C.codegen();
    ASSERT(code.find("4 * i_o + i_i") != std::string::npos);
}

// --- test_swap ---
void test_swap() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();

    C.swap(j, k);  // j ↔ k
    ASSERT(C.loop_order_[0]->name() == "i");
    ASSERT(C.loop_order_[1]->name() == "k");
    ASSERT(C.loop_order_[2]->name() == "j");
}

// --- test_reorder ---
void test_reorder() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();

    C.reorder({k, i, j});
    ASSERT(C.loop_order_[0]->name() == "k");
    ASSERT(C.loop_order_[1]->name() == "i");
    ASSERT(C.loop_order_[2]->name() == "j");
}

// --- test_tile ---
void test_tile() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();

    auto [i_o, j_o, i_i, j_i] = C.tile(i, j, 4, 4);
    ASSERT(i_o->name() == "i_o");
    ASSERT(j_o->name() == "j_o");
    ASSERT(i_i->name() == "i_i");
    ASSERT(j_i->name() == "j_i");

    // tile 後 order: i_o, j_o, k, i_i, j_i
    ASSERT(C.loop_order_.size() == 5);
    ASSERT(C.loop_order_[0]->name() == "i_o");
    ASSERT(C.loop_order_[1]->name() == "j_o");
    ASSERT(C.loop_order_[2]->name() == "k");
    ASSERT(C.loop_order_[3]->name() == "i_i");
    ASSERT(C.loop_order_[4]->name() == "j_i");

    std::string code = C.codegen();
    ASSERT(code.find("4 * i_o + i_i") != std::string::npos);
    ASSERT(code.find("4 * j_o + j_i") != std::string::npos);
}

// --- test_full_tile_6loops ---
void test_full_tile_6loops() {
    // 完整六層 tiled matmul: i_o, j_o, k_o, i_i, j_i, k_i
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();

    auto [i_o, i_i] = i->split(4);
    auto [j_o, j_i] = j->split(4);
    auto [k_o, k_i] = k->split(4);

    // reorder 成標準六層: i_o, j_o, k_o, i_i, j_i, k_i
    C.reorder({i_o, j_o, k_o, i_i, j_i, k_i});

    ASSERT(C.loop_order_.size() == 6);
    ASSERT(C.loop_order_[0]->name() == "i_o");
    ASSERT(C.loop_order_[1]->name() == "j_o");
    ASSERT(C.loop_order_[2]->name() == "k_o");
    ASSERT(C.loop_order_[3]->name() == "i_i");
    ASSERT(C.loop_order_[4]->name() == "j_i");
    ASSERT(C.loop_order_[5]->name() == "k_i");

    std::string code = C.codegen();
    // 六層迴圈都存在
    ASSERT(code.find("for (int i_o") != std::string::npos);
    ASSERT(code.find("for (int j_o") != std::string::npos);
    ASSERT(code.find("for (int k_o") != std::string::npos);
    ASSERT(code.find("for (int i_i") != std::string::npos);
    ASSERT(code.find("for (int j_i") != std::string::npos);
    ASSERT(code.find("for (int k_i") != std::string::npos);
}

// --- test_parallel ---
void test_parallel() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();
    i->parallel();

    std::string code = C.codegen();
    ASSERT(code.find("#pragma omp parallel for") != std::string::npos);
}

// --- test_vectorize ---
void test_vectorize() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;
    auto [i, j, k] = C.loops();
    auto [j_o, j_i] = j->split(4);
    j_i->vectorize();

    std::string code = C.codegen();
    ASSERT(code.find("#pragma omp simd") != std::string::npos);
}

// --- test_kernel_correctness ---
void test_kernel_correctness() {
    // 用 codegen_with_config 生成 kernel，編譯執行驗證正確性
    const int M = 8, K = 4, N = 8;

    Tensor tA("A", M, K);
    Tensor tB("B", K, N);
    auto expr = tA * tB;

    // 測試多種 config
    Expr::TileConfig configs[] = {
        { 2, 2, 2, Expr::TileConfig::IJK },
        { 4, 4, 2, Expr::TileConfig::IKJ },
        { 8, 8, 4, Expr::TileConfig::KIJ },
    };

    for (auto& cfg : configs) {
        std::string kernel = expr.codegen_with_config(cfg);

        std::ostringstream prog;
        prog << "#include <iostream>\n#include <cmath>\n#include <cstdlib>\n\n"
             << "int main() {\n"
             << "    const int M=" << M << ",K=" << K << ",N=" << N << ";\n"
             << "    double A[M*K], B[K*N], C[M*N], C_ref[M*N];\n"
             << "    srand(42);\n"
             << "    for(int i=0;i<M*K;i++) A[i]=(double)(rand()%100)/10.0;\n"
             << "    for(int i=0;i<K*N;i++) B[i]=(double)(rand()%100)/10.0;\n"
             << "    // reference\n"
             << "    for(int i=0;i<M;i++) for(int j=0;j<N;j++) {\n"
             << "        C_ref[i*N+j]=0;\n"
             << "        for(int k=0;k<K;k++) C_ref[i*N+j]+=A[i*K+k]*B[k*N+j];\n"
             << "    }\n"
             << "    // generated kernel\n"
             << kernel << "\n"
             << "    for(int i=0;i<M*N;i++) {\n"
             << "        if(std::abs(C[i]-C_ref[i])>1e-6) { std::cout<<0; return 1; }\n"
             << "    }\n"
             << "    std::cout<<1; return 0;\n}\n";

        std::string tag = std::to_string(cfg.ti) + "_" + std::to_string(cfg.tj)
                        + "_" + std::to_string(cfg.tk) + "_" + std::to_string((int)cfg.reorder);
        std::string src = "/tmp/namiloop_tc_" + tag + ".cpp";
        std::string bin = "/tmp/namiloop_tc_" + tag;
        { std::ofstream f(src); f << prog.str(); }

        int ret = std::system(("g++ -std=c++17 -O2 -o " + bin + " " + src + " 2>&1").c_str());
        ASSERT(ret == 0);

        FILE* pipe = popen(bin.c_str(), "r");
        ASSERT(pipe);
        char buf[64];
        ASSERT(fgets(buf, sizeof(buf), pipe));
        pclose(pipe);
        ASSERT(std::atoi(buf) == 1);

        std::remove(src.c_str());
        std::remove(bin.c_str());
    }
}

// --- test_codegen_via_schedule ---
void test_codegen_via_schedule() {
    // 用排程 API 生成 kernel，驗證正確性
    const int M = 8, K = 4, N = 8;
    Tensor tA("A", M, K);
    Tensor tB("B", K, N);
    auto C = tA * tB;
    auto [i, j, k] = C.loops();

    auto [i_o, i_i] = i->split(4);
    auto [j_o, j_i] = j->split(4);
    i_o->parallel();

    std::string kernel = C.codegen();

    std::ostringstream prog;
    prog << "#include <iostream>\n#include <cmath>\n#include <cstdlib>\n\n"
         << "int main() {\n"
         << "    const int M=" << M << ",K=" << K << ",N=" << N << ";\n"
         << "    double A[M*K], B[K*N], C[M*N], C_ref[M*N];\n"
         << "    srand(42);\n"
         << "    for(int i=0;i<M*K;i++) A[i]=(double)(rand()%100)/10.0;\n"
         << "    for(int i=0;i<K*N;i++) B[i]=(double)(rand()%100)/10.0;\n"
         << "    for(int i=0;i<M;i++) for(int j=0;j<N;j++) {\n"
         << "        C_ref[i*N+j]=0;\n"
         << "        for(int k=0;k<K;k++) C_ref[i*N+j]+=A[i*K+k]*B[k*N+j];\n"
         << "    }\n"
         << kernel << "\n"
         << "    for(int i=0;i<M*N;i++) {\n"
         << "        if(std::abs(C[i]-C_ref[i])>1e-6) { std::cout<<0; return 1; }\n"
         << "    }\n"
         << "    std::cout<<1; return 0;\n}\n";

    std::string src = "/tmp/namiloop_tc_sched.cpp";
    std::string bin = "/tmp/namiloop_tc_sched";
    { std::ofstream f(src); f << prog.str(); }

    int ret = std::system(("g++ -std=c++17 -O2 -o " + bin + " " + src + " 2>&1").c_str());
    ASSERT(ret == 0);

    FILE* pipe = popen(bin.c_str(), "r");
    ASSERT(pipe);
    char buf[64];
    ASSERT(fgets(buf, sizeof(buf), pipe));
    pclose(pipe);
    ASSERT(std::atoi(buf) == 1);

    std::remove(src.c_str());
    std::remove(bin.c_str());
}

int main() {
    std::cout << "\n🧪 NamiLoop 測試\n";
    std::cout << "==================\n";

    RUN_TEST(test_tensor_create);
    RUN_TEST(test_expr_matmul);
    RUN_TEST(test_loops_api);
    RUN_TEST(test_split);
    RUN_TEST(test_swap);
    RUN_TEST(test_reorder);
    RUN_TEST(test_tile);
    RUN_TEST(test_full_tile_6loops);
    RUN_TEST(test_parallel);
    RUN_TEST(test_vectorize);
    RUN_TEST(test_kernel_correctness);
    RUN_TEST(test_codegen_via_schedule);

    std::cout << "\n==================\n";
    std::cout << "✅ " << tests_passed << "/12 tests passed\n\n";

    return (tests_passed == 12) ? 0 : 1;
}
