// ============================================================================
// NamiLoop — 自動化迴圈優化 DSL Framework
// Authors: Ryan & Nami
// License: MIT
// 
// 靈感來源：Halide / Tiramisu / Ryan's autoloop
// 設計理念：演算法與排程分離，Expression Template 語法糖
// ============================================================================
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <functional>
#include <iomanip>

namespace namiloop {

// ============================================================================
// 仿射表達式 ax + b（致敬 Ryan 原版 autoloop 的 ax_b 設計）
// ============================================================================
struct AffineExpr {
    std::string a = "1";   // 係數
    std::string x = "";    // 變數名
    std::string b = "0";   // 常數偏移

    // 生成表達式字串
    std::string gen() const {
        if (x.empty()) return b;
        if (a == "1" && b == "0") return x;
        if (a == "1") return x + " + " + b;
        if (b == "0") return a + " * " + x;
        return a + " * " + x + " + " + b;
    }
};

// ============================================================================
// 迴圈維度資訊
// ============================================================================
struct LoopDim {
    std::string name;       // 迴圈變數名
    int min = 0;            // 起始值
    int max = 0;            // 結束值（exclusive）
    int step = 1;           // 步幅
    bool is_parallel = false;
    AffineExpr index_expr;  // 原始索引的仿射表達式

    LoopDim() = default;
    LoopDim(const std::string& n, int lo, int hi, int s = 1)
        : name(n), min(lo), max(hi), step(s) {
        index_expr.x = n;
        index_expr.a = "1";
        index_expr.b = "0";
    }
};

// ============================================================================
// 排程指令
// ============================================================================
enum class SchedOp { SPLIT, REORDER, TILE, PARALLEL };

struct SchedCmd {
    SchedOp op;
    int dim1 = 0;
    int dim2 = 0;
    int factor1 = 0;
    int factor2 = 0;
};

// ============================================================================
// Tensor — 記錄矩陣的名稱和維度
// ============================================================================
class Tensor {
public:
    std::string name_;
    int rows_, cols_;

    Tensor() : name_("?"), rows_(0), cols_(0) {}
    Tensor(const std::string& name, int rows, int cols)
        : name_(name), rows_(rows), cols_(cols) {}

    const std::string& name() const { return name_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
};

// ============================================================================
// 前向宣告
// ============================================================================
class Expr;

// ============================================================================
// Auto-Tile 搜索結果
// ============================================================================
struct TileResult {
    int tile_i = 0;
    int tile_j = 0;
    double time_ms = 0.0;
};

struct AutoTileReport {
    std::vector<TileResult> results;
    TileResult best;
    std::string kernel_code;
    int M = 0, K = 0, N = 0;

    // 印出搜索報告
    void print_report() const {
        std::cout << "\n🔍 Auto-Tile Search: "
                  << "A(" << M << "," << K << ") × B(" << K << "," << N << ")\n";
        for (auto& r : results) {
            std::cout << "  tile(" << r.tile_i << "," << r.tile_j << ")"
                      << std::string(8 - std::to_string(r.tile_i).size() 
                                       - std::to_string(r.tile_j).size(), ' ')
                      << " → " << std::fixed << std::setprecision(1)
                      << r.time_ms << "ms";
            if (r.tile_i == best.tile_i && r.tile_j == best.tile_j) {
                std::cout << "  ← 🏆 Best!";
            }
            std::cout << "\n";
        }
        std::cout << "\n🏆 Best config: tile(" << best.tile_i << "," << best.tile_j
                  << ") → " << std::fixed << std::setprecision(1) << best.time_ms << "ms\n";
    }

    // 儲存最佳 kernel
    void save(const std::string& path) const {
        std::ofstream out(path);
        out << kernel_code;
        out.close();
        std::cout << "📄 Saved to: " << path << "\n";
    }
};

// ============================================================================
// Expr — AST 節點（Expression Template 核心）
// ============================================================================
class Expr {
public:
    enum OpType { NONE, MATMUL };

    OpType op_;
    std::shared_ptr<Tensor> lhs_;
    std::shared_ptr<Tensor> rhs_;
    
    // 排程狀態
    std::vector<LoopDim> loops_;
    std::vector<SchedCmd> schedule_;
    bool has_parallel_ = false;
    int parallel_dim_ = 0;

    Expr() : op_(NONE) {}

    Expr(OpType op, const Tensor& lhs, const Tensor& rhs)
        : op_(op),
          lhs_(std::make_shared<Tensor>(lhs)),
          rhs_(std::make_shared<Tensor>(rhs)) {
        // matmul: C[i][j] += A[i][k] * B[k][j]
        // 預設迴圈順序: i, j, k
        if (op == MATMUL) {
            loops_.emplace_back("i", 0, lhs.rows());
            loops_.emplace_back("j", 0, rhs.cols());
            loops_.emplace_back("k", 0, lhs.cols());
        }
    }

    // ================================================================
    // 排程 API
    // ================================================================

    // split(dim, factor) — 分割迴圈
    Expr& split(int dim, int factor) {
        assert(dim >= 0 && dim < (int)loops_.size());
        SchedCmd cmd;
        cmd.op = SchedOp::SPLIT;
        cmd.dim1 = dim;
        cmd.factor1 = factor;
        schedule_.push_back(cmd);
        return *this;
    }

    // reorder(dim1, dim2) — 交換迴圈
    Expr& reorder(int dim1, int dim2) {
        SchedCmd cmd;
        cmd.op = SchedOp::REORDER;
        cmd.dim1 = dim1;
        cmd.dim2 = dim2;
        schedule_.push_back(cmd);
        return *this;
    }

    // tile(ti, tj) — 分塊（對 dim0 和 dim1 做 split + reorder）
    Expr& tile(int ti, int tj) {
        SchedCmd cmd;
        cmd.op = SchedOp::TILE;
        cmd.factor1 = ti;
        cmd.factor2 = tj;
        schedule_.push_back(cmd);
        return *this;
    }

    // parallel(dim) — 標記並行
    Expr& parallel(int dim = 0) {
        has_parallel_ = true;
        parallel_dim_ = dim;
        SchedCmd cmd;
        cmd.op = SchedOp::PARALLEL;
        cmd.dim1 = dim;
        schedule_.push_back(cmd);
        return *this;
    }

    // ================================================================
    // CodeGen — 生成 C 迴圈代碼
    // ================================================================
    std::string codegen_matmul(int ti, int tj, bool do_parallel = false) const {
        assert(op_ == MATMUL);
        int M = lhs_->rows();
        int K = lhs_->cols();
        int N = rhs_->cols();

        std::ostringstream out;
        out << "// NamiLoop 生成的 matmul kernel\n";
        out << "// " << lhs_->name() << "(" << M << "," << K << ") × "
            << rhs_->name() << "(" << K << "," << N << ")\n";
        out << "// tile(" << ti << "," << tj << ")\n";

        std::string indent = "";
        auto emit = [&](const std::string& s) {
            out << indent << s << "\n";
        };

        // 是否需要 tiling
        bool tiled = (ti > 1 || tj > 1);

        if (tiled) {
            // 外層迴圈: i_o, j_o
            if (do_parallel) emit("#pragma omp parallel for collapse(2)");
            emit("for (int i_o = 0; i_o < " + std::to_string(M / ti) + "; i_o++) {");
            indent = "    ";
            emit("for (int j_o = 0; j_o < " + std::to_string(N / tj) + "; j_o++) {");
            indent = "        ";
            
            // 內層迴圈: i_i, j_i
            emit("for (int i_i = 0; i_i < " + std::to_string(ti) + "; i_i++) {");
            indent = "            ";
            emit("for (int j_i = 0; j_i < " + std::to_string(tj) + "; j_i++) {");
            indent = "                ";

            // 仿射索引（ax + b 風格）
            emit("int i = " + std::to_string(ti) + " * i_o + i_i;");
            emit("int j = " + std::to_string(tj) + " * j_o + j_i;");
            emit("C[i * " + std::to_string(N) + " + j] = 0;");

            // k 迴圈
            emit("for (int k = 0; k < " + std::to_string(K) + "; k++) {");
            indent = "                    ";
            emit("C[i * " + std::to_string(N) + " + j] += "
                 "A[i * " + std::to_string(K) + " + k] * "
                 "B[k * " + std::to_string(N) + " + j];");
            indent = "                ";
            emit("}");

            // 關閉迴圈
            indent = "            "; emit("}");
            indent = "        "; emit("}");
            indent = "    "; emit("}");
            indent = ""; emit("}");
        } else {
            // 無 tiling 的基本迴圈
            if (do_parallel) emit("#pragma omp parallel for");
            emit("for (int i = 0; i < " + std::to_string(M) + "; i++) {");
            indent = "    ";
            emit("for (int j = 0; j < " + std::to_string(N) + "; j++) {");
            indent = "        ";
            emit("C[i * " + std::to_string(N) + " + j] = 0;");
            emit("for (int k = 0; k < " + std::to_string(K) + "; k++) {");
            indent = "            ";
            emit("C[i * " + std::to_string(N) + " + j] += "
                 "A[i * " + std::to_string(K) + " + k] * "
                 "B[k * " + std::to_string(N) + " + j];");
            indent = "        ";
            emit("}");
            indent = "    "; emit("}");
            indent = ""; emit("}");
        }

        return out.str();
    }

    // 根據當前排程生成 codegen
    std::string codegen() const {
        if (op_ == MATMUL) {
            int ti = 1, tj = 1;
            bool par = has_parallel_;
            for (auto& cmd : schedule_) {
                if (cmd.op == SchedOp::TILE) {
                    ti = cmd.factor1;
                    tj = cmd.factor2;
                } else if (cmd.op == SchedOp::SPLIT && cmd.dim1 == 0) {
                    ti = cmd.factor1;
                } else if (cmd.op == SchedOp::SPLIT && cmd.dim1 == 1) {
                    tj = cmd.factor1;
                }
            }
            return codegen_matmul(ti, tj, par);
        }
        return "// unsupported op\n";
    }

    // 寫入檔案
    void codegen(const std::string& path) const {
        std::ofstream out(path);
        out << codegen();
        out.close();
    }

    // ================================================================
    // Auto-Tile 搜索
    // ================================================================
    AutoTileReport auto_tile(int benchmark_runs = 5) const {
        assert(op_ == MATMUL);
        int M = lhs_->rows();
        int K = lhs_->cols();
        int N = rhs_->cols();

        // 候選 tile sizes
        std::vector<int> candidates = {1, 2, 4, 8, 16, 32};

        AutoTileReport report;
        report.M = M;
        report.K = K;
        report.N = N;
        report.best.time_ms = 1e18;

        for (int ti : candidates) {
            for (int tj : candidates) {
                // 跳過不能整除的
                if (M % ti != 0 || N % tj != 0) continue;
                // 跳過 tile > 維度的
                if (ti > M || tj > N) continue;

                // 1. codegen
                std::string kernel = codegen_matmul(ti, tj, false);

                // 2. 生成完整的 benchmark 程式
                std::string prog = generate_benchmark_program(kernel, M, K, N, benchmark_runs);

                // 3. 寫入暫存檔
                std::string src_path = "/tmp/namiloop_bench_" + std::to_string(ti) + "_" + std::to_string(tj) + ".cpp";
                std::string bin_path = "/tmp/namiloop_bench_" + std::to_string(ti) + "_" + std::to_string(tj);
                {
                    std::ofstream f(src_path);
                    f << prog;
                }

                // 4. 編譯
                std::string compile_cmd = "g++ -std=c++17 -O2 -o " + bin_path + " " + src_path + " 2>/dev/null";
                int ret = std::system(compile_cmd.c_str());
                if (ret != 0) continue;

                // 5. 執行 benchmark
                std::string run_cmd = bin_path + " 2>/dev/null";
                FILE* pipe = popen(run_cmd.c_str(), "r");
                if (!pipe) continue;

                char buf[256];
                double time_ms = 1e18;
                if (fgets(buf, sizeof(buf), pipe)) {
                    time_ms = std::atof(buf);
                }
                pclose(pipe);

                // 6. 清理
                std::remove(src_path.c_str());
                std::remove(bin_path.c_str());

                TileResult tr;
                tr.tile_i = ti;
                tr.tile_j = tj;
                tr.time_ms = time_ms;
                report.results.push_back(tr);

                if (time_ms < report.best.time_ms) {
                    report.best = tr;
                    report.kernel_code = kernel;
                }
            }
        }

        // 按時間排序
        std::sort(report.results.begin(), report.results.end(),
                  [](const TileResult& a, const TileResult& b) {
                      return a.time_ms < b.time_ms;
                  });

        return report;
    }

private:
    // 生成 benchmark 程式碼
    std::string generate_benchmark_program(const std::string& kernel,
                                           int M, int K, int N,
                                           int runs) const {
        std::ostringstream out;
        out << "#include <chrono>\n";
        out << "#include <iostream>\n";
        out << "#include <vector>\n";
        out << "#include <algorithm>\n";
        out << "#include <cstdlib>\n\n";
        out << "int main() {\n";
        out << "    const int M = " << M << ", K = " << K << ", N = " << N << ";\n";
        out << "    std::vector<double> A(M * K), B(K * N), C(M * N);\n";
        out << "    // 初始化隨機資料\n";
        out << "    for (int i = 0; i < M * K; i++) A[i] = (double)(rand() % 100) / 100.0;\n";
        out << "    for (int i = 0; i < K * N; i++) B[i] = (double)(rand() % 100) / 100.0;\n";
        out << "    double* A_ptr = A.data();\n";
        out << "    double* B_ptr = B.data();\n";
        out << "    double* C_ptr = C.data();\n\n";
        out << "    // 用 macro 讓 kernel 能直接存取\n";
        out << "    #define A(r,c) A_ptr[(r) * K + (c)]\n";
        out << "    #define B(r,c) B_ptr[(r) * N + (c)]\n\n";
        out << "    std::vector<double> times;\n";
        out << "    for (int run = 0; run < " << runs << "; run++) {\n";
        out << "        auto t0 = std::chrono::high_resolution_clock::now();\n";
        out << "        // === kernel begin ===\n";
        out << "        {\n";
        out << "            double* A = A_ptr;\n";
        out << "            double* B = B_ptr;\n";
        out << "            double* C = C_ptr;\n";
        out << kernel;
        out << "        }\n";
        out << "        // === kernel end ===\n";
        out << "        auto t1 = std::chrono::high_resolution_clock::now();\n";
        out << "        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();\n";
        out << "        times.push_back(ms);\n";
        out << "    }\n";
        out << "    std::sort(times.begin(), times.end());\n";
        out << "    // 取中位數\n";
        out << "    double median = times[times.size() / 2];\n";
        out << "    std::cout << median << std::endl;\n";
        out << "    return 0;\n";
        out << "}\n";
        return out.str();
    }
};

// ============================================================================
// operator* — 建立 matmul AST（不計算，只記錄）
// ============================================================================
inline Expr operator*(const Tensor& lhs, const Tensor& rhs) {
    assert(lhs.cols() == rhs.rows() && "矩陣維度不匹配！");
    return Expr(Expr::MATMUL, lhs, rhs);
}

} // namespace namiloop
