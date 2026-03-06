// ============================================================================
// NamiLoop — 自動化迴圈優化 DSL Framework (v2)
// Authors: Ryan & Nami
// License: MIT
// 
// 靈感來源：Halide / Tiramisu / Ryan's autoloop
// 設計理念：演算法與排程分離，LoopVar 物件操作排程
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
#include <tuple>
#include <utility>
#include <set>

namespace namiloop {

// ============================================================================
// LoopVar — 迴圈變數物件（Halide 風格）
// ============================================================================
class LoopVar {
public:
    std::string name_;          // "i", "i_o", "i_i"
    int min_ = 0;
    int max_ = 0;               // exclusive
    int factor_ = 0;            // split factor (0 = 未 split)
    bool is_parallel_ = false;
    LoopVar* parent_ = nullptr; // split 前的原始 var

    // 用於在 Expr 中找到自己的 ID
    int id_ = -1;

    // 所屬的 Expr（反向指標，用於 split/parallel 操作）
    class Expr* owner_ = nullptr;

    LoopVar() = default;
    LoopVar(const std::string& name, int lo, int hi)
        : name_(name), min_(lo), max_(hi) {}

    const std::string& name() const { return name_; }
    int extent() const { return max_ - min_; }

    // split 回傳 pair<outer, inner>
    std::pair<LoopVar, LoopVar> split(int factor);

    // 標記平行化
    void parallel() {
        is_parallel_ = true;
        // 同步到 owner 的 loops_ 裡
        if (owner_ && id_ >= 0) {
            owner_->set_parallel(id_);
        }
    }

    // 比較用（用 id）
    bool operator==(const LoopVar& o) const { return id_ == o.id_ && name_ == o.name_; }
    bool operator!=(const LoopVar& o) const { return !(*this == o); }
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
// Auto-Tile 搜索結果（v2: 含 loop order）
// ============================================================================
struct TileResult {
    int tile_i = 0;
    int tile_j = 0;
    int tile_k = 0;             // k 也可以 tile
    std::vector<std::string> loop_order;  // 迴圈順序名稱
    double time_ms = 0.0;

    std::string order_str() const {
        std::string s = "[";
        for (size_t i = 0; i < loop_order.size(); i++) {
            if (i > 0) s += ",";
            s += loop_order[i];
        }
        s += "]";
        return s;
    }
};

struct AutoTileReport {
    std::vector<TileResult> results;
    TileResult best;
    std::string kernel_code;
    int M = 0, K = 0, N = 0;

    void print_report() const {
        std::cout << "\n🔍 Auto-Tile Search: "
                  << "A(" << M << "," << K << ") × B(" << K << "," << N << ")\n";
        for (auto& r : results) {
            std::cout << "  tile(" << r.tile_i << "," << r.tile_j << "," << r.tile_k << ")"
                      << "  order" << r.order_str()
                      << " → " << std::fixed << std::setprecision(1)
                      << r.time_ms << "ms";
            if (r.tile_i == best.tile_i && r.tile_j == best.tile_j &&
                r.tile_k == best.tile_k && r.loop_order == best.loop_order) {
                std::cout << "  ← 🏆";
            }
            std::cout << "\n";
        }
        std::cout << "\n🏆 Best: tile(" << best.tile_i << "," << best.tile_j
                  << "," << best.tile_k << ") order" << best.order_str()
                  << " → " << std::fixed << std::setprecision(1) << best.time_ms << "ms\n";
    }

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

    // 迴圈狀態（LoopVar 列表 = 當前的迴圈巢狀順序）
    std::vector<LoopVar> loops_;

    // 記錄哪些原始維度被 split 了，用於 codegen 時計算仿射索引
    // key: 原始 loop 名稱 ("i","j","k")
    // value: {outer_name, inner_name, factor}
    struct SplitInfo {
        std::string orig;
        std::string outer;
        std::string inner;
        int factor;
    };
    std::vector<SplitInfo> splits_;

    int next_id_ = 0; // LoopVar id 分配器

    Expr() : op_(NONE) {}

    Expr(OpType op, const Tensor& lhs, const Tensor& rhs)
        : op_(op),
          lhs_(std::make_shared<Tensor>(lhs)),
          rhs_(std::make_shared<Tensor>(rhs)) {
        if (op == MATMUL) {
            add_loop("i", 0, lhs.rows());
            add_loop("j", 0, rhs.cols());
            add_loop("k", 0, lhs.cols());
        }
    }

    // ================================================================
    // 取得迴圈變數（structured bindings 用）
    // ================================================================
    std::tuple<LoopVar, LoopVar, LoopVar> loops() {
        assert(loops_.size() >= 3);
        // 回傳的 LoopVar 帶有 owner 反向指標
        auto a = loops_[0]; a.owner_ = this;
        auto b = loops_[1]; b.owner_ = this;
        auto c = loops_[2]; c.owner_ = this;
        return {a, b, c};
    }

    // ================================================================
    // LoopVar-based 排程 API
    // ================================================================

    // split：在 loops_ 裡找到 var，替換成 outer + inner
    std::pair<LoopVar, LoopVar> split_var(const LoopVar& var, int factor) {
        int pos = find_loop(var);
        assert(pos >= 0 && "LoopVar not found in loops_");
        auto& orig = loops_[pos];
        int extent = orig.max_ - orig.min_;
        assert(extent % factor == 0 && "split factor 必須整除迴圈範圍");

        std::string oname = orig.name_ + "_o";
        std::string iname = orig.name_ + "_i";

        // 記錄 split 資訊
        SplitInfo si;
        si.orig = orig.name_;
        si.outer = oname;
        si.inner = iname;
        si.factor = factor;
        splits_.push_back(si);

        // 建立 outer 和 inner
        LoopVar outer(oname, 0, extent / factor);
        outer.id_ = next_id_++;
        outer.owner_ = this;
        outer.factor_ = factor;

        LoopVar inner(iname, 0, factor);
        inner.id_ = next_id_++;
        inner.owner_ = this;
        inner.factor_ = factor;

        // 替換：原位變成 outer，後面插入 inner
        loops_[pos] = outer;
        loops_.insert(loops_.begin() + pos + 1, inner);

        return {outer, inner};
    }

    // swap 兩個 LoopVar
    void swap(const LoopVar& a, const LoopVar& b) {
        int pa = find_loop(a);
        int pb = find_loop(b);
        assert(pa >= 0 && pb >= 0 && "LoopVar not found");
        std::swap(loops_[pa], loops_[pb]);
    }

    // reorder：完整指定迴圈順序
    void reorder(const std::vector<LoopVar>& order) {
        assert(order.size() == loops_.size() && "reorder 必須包含所有 LoopVar");
        std::vector<LoopVar> new_loops;
        for (auto& v : order) {
            int pos = find_loop(v);
            assert(pos >= 0 && "LoopVar not found");
            new_loops.push_back(loops_[pos]);
        }
        loops_ = new_loops;
    }

    // 舊式 API 保留（向後相容）
    Expr& split(int dim, int factor) {
        assert(dim >= 0 && dim < (int)loops_.size());
        split_var(loops_[dim], factor);
        return *this;
    }

    Expr& tile(int ti, int tj) {
        // split i (dim 0), split j (dim 1 → 現在位置可能是 1 或 2)
        auto [i_o, i_i] = split_var(loops_[0], ti);
        // j 現在在位置 2（因為 split i 插入了 i_i）
        auto [j_o, j_i] = split_var(loops_[2], tj);
        // reorder: i_o, j_o, i_i, j_i, k
        // 找到 k
        LoopVar k_var;
        for (auto& l : loops_) {
            if (l.name_ == "k") { k_var = l; break; }
        }
        reorder({i_o, j_o, i_i, j_i, k_var});
        return *this;
    }

    Expr& parallel(int dim = 0) {
        assert(dim >= 0 && dim < (int)loops_.size());
        loops_[dim].is_parallel_ = true;
        return *this;
    }

    // 給 LoopVar::parallel() 用的
    void set_parallel(int id) {
        for (auto& l : loops_) {
            if (l.id_ == id) {
                l.is_parallel_ = true;
                return;
            }
        }
    }

    // ================================================================
    // CodeGen v2 — 根據 loops_ 順序生成任意巢狀迴圈
    // ================================================================
    std::string codegen() const {
        if (op_ != MATMUL) return "// unsupported op\n";

        int M = lhs_->rows();
        int K = lhs_->cols();
        int N = rhs_->cols();

        std::ostringstream out;
        out << "// NamiLoop 生成的 matmul kernel\n";
        out << "// " << lhs_->name() << "(" << M << "," << K << ") × "
            << rhs_->name() << "(" << K << "," << N << ")\n";

        // 列出 loop order
        out << "// loop order: ";
        for (size_t i = 0; i < loops_.size(); i++) {
            if (i > 0) out << ", ";
            out << loops_[i].name_;
        }
        out << "\n";

        // 生成迴圈
        std::string indent = "";
        auto emit = [&](const std::string& s) { out << indent << s << "\n"; };

        // 判斷是否有 split
        bool has_splits = !splits_.empty();

        // 開迴圈
        for (size_t d = 0; d < loops_.size(); d++) {
            auto& l = loops_[d];
            if (l.is_parallel_) {
                emit("#pragma omp parallel for");
            }
            emit("for (int " + l.name_ + " = " + std::to_string(l.min_) +
                 "; " + l.name_ + " < " + std::to_string(l.max_) +
                 "; " + l.name_ + "++) {");
            indent += "    ";
        }

        // 計算仿射索引
        // 找出原始 i, j, k 的表達式
        auto index_expr = [&](const std::string& orig) -> std::string {
            for (auto& s : splits_) {
                if (s.orig == orig) {
                    return std::to_string(s.factor) + " * " + s.outer + " + " + s.inner;
                }
            }
            return orig; // 沒 split，直接用原始變數
        };

        std::string i_expr = index_expr("i");
        std::string j_expr = index_expr("j");
        std::string k_expr = index_expr("k");

        if (has_splits) {
            // 發射仿射索引計算
            if (i_expr != "i") emit("int i = " + i_expr + ";");
            if (j_expr != "j") emit("int j = " + j_expr + ";");
            if (k_expr != "k") emit("int k = " + k_expr + ";");
        }

        // 找到最內層的非 k 迴圈位置，在那之前歸零 C
        // 策略：在 k 迴圈外面歸零 C
        // 但因為 loop order 任意，需要更聰明的策略
        // 簡單做法：找第一個 k 相關的迴圈，在它前面 emit C[i][j] = 0
        // 更簡單：用 += 前先清零的方式 — 在最外面初始化 C = 0
        // 最簡單且正確：不在 kernel 裡清零，假設 C 已經初始化為 0
        // （跟 v1 一致，v1 的 tiled 版本在 k 迴圈前有 C[i][j]=0）
        
        // 實際上需要在 k 迴圈外面歸零。找到 k 相關迴圈的位置
        // 如果 k 沒被 split，找 "k" 的位置
        // 如果 k 被 split，找 k_o 的位置（外層 k）
        // 在那個位置的 indent level emit C = 0
        
        // 為了簡單+正確，我們在 kernel 內不歸零 C，
        // 而是在 benchmark 程式裡初始化。用 += 就好。
        // 但這跟 v1 行為不同... v1 在 kernel 裡有 C[i][j] = 0
        
        // 折衷：在所有 k 相關迴圈之前、所有 i/j 迴圈之後 emit C=0
        // 這需要分析 loop order... 直接做吧

        // 不對，我們直接用正確的做法：
        // 找到 k 相關的最外層迴圈在 loops_ 的位置 k_pos
        // 在 k_pos 的 indent level（已經在 k 外面了）emit C=0
        // 這表示我們需要在開迴圈時，在適當位置插入 C=0
        
        // 重寫：用兩段式 emit

        // 算了，重新用乾淨的方式重寫 codegen
        out.str(""); // 清空重來
        out << "// NamiLoop 生成的 matmul kernel\n";
        out << "// " << lhs_->name() << "(" << M << "," << K << ") × "
            << rhs_->name() << "(" << K << "," << N << ")\n";
        out << "// loop order: ";
        for (size_t ii = 0; ii < loops_.size(); ii++) {
            if (ii > 0) out << ", ";
            out << loops_[ii].name_;
        }
        out << "\n";

        return codegen_v2(out.str(), M, K, N);
    }

    void codegen(const std::string& path) const {
        std::ofstream f(path);
        f << codegen();
        f.close();
    }

    // ================================================================
    // 靜態 codegen：給定 tile sizes 和 loop order 生成 kernel
    // ================================================================
    static std::string codegen_with_config(
        int M, int K, int N,
        int ti, int tj, int tk,
        const std::vector<std::string>& loop_order,
        bool do_parallel = false,
        const std::string& A_name = "A",
        const std::string& B_name = "B")
    {
        std::ostringstream out;
        out << "// NamiLoop 生成的 matmul kernel\n";
        out << "// " << A_name << "(" << M << "," << K << ") × "
            << B_name << "(" << K << "," << N << ")\n";
        out << "// tile(" << ti << "," << tj << "," << tk << ") order[";
        for (size_t i = 0; i < loop_order.size(); i++) {
            if (i > 0) out << ",";
            out << loop_order[i];
        }
        out << "]\n";

        // 建立迴圈維度表
        struct DimInfo {
            std::string name;
            int lo, hi;
        };
        std::vector<DimInfo> dims;
        for (auto& ln : loop_order) {
            DimInfo d;
            d.name = ln;
            d.lo = 0;
            if (ln == "i_o") d.hi = M / ti;
            else if (ln == "i_i") d.hi = ti;
            else if (ln == "j_o") d.hi = N / tj;
            else if (ln == "j_i") d.hi = tj;
            else if (ln == "k_o") d.hi = K / tk;
            else if (ln == "k_i") d.hi = tk;
            else if (ln == "i") d.hi = M;
            else if (ln == "j") d.hi = N;
            else if (ln == "k") d.hi = K;
            else d.hi = 1;
            dims.push_back(d);
        }

        // 找 k 相關的最外層位置
        int k_outer_pos = -1;
        for (size_t i = 0; i < loop_order.size(); i++) {
            auto& n = loop_order[i];
            if (n == "k" || n == "k_o" || n == "k_i") {
                if (k_outer_pos < 0) k_outer_pos = (int)i;
            }
        }

        std::string indent = "";
        auto emit = [&](const std::string& s) { out << indent << s << "\n"; };

        // emit 迴圈
        for (size_t i = 0; i < dims.size(); i++) {
            if ((int)i == 0 && do_parallel) {
                emit("#pragma omp parallel for");
            }
            emit("for (int " + dims[i].name + " = " + std::to_string(dims[i].lo) +
                 "; " + dims[i].name + " < " + std::to_string(dims[i].hi) +
                 "; " + dims[i].name + "++) {");
            indent += "    ";

            // 在 k 的最外層前面 emit 仿射索引 + C=0
            if ((int)i == k_outer_pos - 1 || (k_outer_pos == 0 && (int)i == 0 && dims[i].name[0] != 'k')) {
                // 不在這裡，改在 k_outer_pos 的時候做
            }
            if ((int)i + 1 == k_outer_pos) {
                // 在進入 k 迴圈之前，emit 索引計算和 C=0
                auto i_expr = (ti > 1) ? (std::to_string(ti) + " * i_o + i_i") : "i";
                auto j_expr = (tj > 1) ? (std::to_string(tj) + " * j_o + j_i") : "j";
                // 只有 i 和 j 的分量都已經在外面才能算
                // 檢查 i/j 相關的迴圈是否都在 k_outer_pos 之前
                bool i_ready = is_var_ready("i", ti, loop_order, k_outer_pos);
                bool j_ready = is_var_ready("j", tj, loop_order, k_outer_pos);
                if (i_ready && j_ready) {
                    if (ti > 1) emit("int i = " + i_expr + ";");
                    if (tj > 1) emit("int j = " + j_expr + ";");
                    emit("C[i * " + std::to_string(N) + " + j] = 0;");
                }
            }
        }

        // 最內層：仿射索引 + accumulate
        auto i_expr = (ti > 1) ? (std::to_string(ti) + " * i_o + i_i") : "i";
        auto j_expr = (tj > 1) ? (std::to_string(tj) + " * j_o + j_i") : "j";
        auto k_expr = (tk > 1) ? (std::to_string(tk) + " * k_o + k_i") : "k";

        // 檢查是否已經在 k 前面 emit 過 i/j 的索引
        bool emitted_outside_k = false;
        {
            bool i_ready = is_var_ready("i", ti, loop_order, (k_outer_pos >= 0 ? k_outer_pos : (int)loop_order.size()));
            bool j_ready = is_var_ready("j", tj, loop_order, (k_outer_pos >= 0 ? k_outer_pos : (int)loop_order.size()));
            emitted_outside_k = (k_outer_pos > 0) && i_ready && j_ready;
        }

        if (!emitted_outside_k) {
            // i/j 沒有完全在 k 外面，在最內層計算所有索引
            if (ti > 1) emit("int i = " + i_expr + ";");
            if (tj > 1) emit("int j = " + j_expr + ";");
            if (tk > 1) emit("int k = " + k_expr + ";");
            emit("C[i * " + std::to_string(N) + " + j] += "
                 "A[i * " + std::to_string(K) + " + k] * "
                 "B[k * " + std::to_string(N) + " + j];");
        } else {
            // i/j 已在 k 外面 emit，這裡只需要 k 索引
            if (tk > 1) emit("int k = " + k_expr + ";");
            emit("C[i * " + std::to_string(N) + " + j] += "
                 "A[i * " + std::to_string(K) + " + k] * "
                 "B[k * " + std::to_string(N) + " + j];");
        }

        // 關閉迴圈
        for (size_t i = 0; i < dims.size(); i++) {
            indent.resize(indent.size() - 4);
            out << indent << "}\n";
        }

        return out.str();
    }

    // ================================================================
    // Auto-Tile v2：搜索 tile size + loop order
    // ================================================================
    AutoTileReport auto_tile(int benchmark_runs = 5) const {
        assert(op_ == MATMUL);
        int M = lhs_->rows();
        int K = lhs_->cols();
        int N = rhs_->cols();

        // 候選 tile sizes
        std::vector<int> tile_ij_cands = {1, 2, 4, 8, 16, 32};
        std::vector<int> tile_k_cands = {1, 4, 8, 16};

        // 候選 loop orders（以 name 表示）
        using Order = std::vector<std::string>;
        auto make_orders = [](int ti, int tj, int tk) -> std::vector<Order> {
            std::vector<Order> orders;
            if (ti > 1 && tj > 1 && tk > 1) {
                // 全部都 split
                orders.push_back({"i_o","j_o","k_o","i_i","j_i","k_i"});
                orders.push_back({"i_o","k_o","j_o","i_i","j_i","k_i"});
                orders.push_back({"j_o","i_o","k_o","j_i","i_i","k_i"});
            } else if (ti > 1 && tj > 1 && tk == 1) {
                // i,j split, k 不 split
                orders.push_back({"i_o","j_o","k","i_i","j_i"});
                orders.push_back({"i_o","j_o","i_i","j_i","k"});
                orders.push_back({"i_o","k","j_o","i_i","j_i"});
                orders.push_back({"j_o","i_o","k","j_i","i_i"});
            } else if (ti > 1 && tj == 1 && tk == 1) {
                orders.push_back({"i_o","j","k","i_i"});
                orders.push_back({"i_o","j","i_i","k"});
            } else if (ti == 1 && tj > 1 && tk == 1) {
                orders.push_back({"i","j_o","k","j_i"});
            } else if (ti == 1 && tj == 1 && tk > 1) {
                orders.push_back({"i","j","k_o","k_i"});
            } else if (ti > 1 && tj == 1 && tk > 1) {
                orders.push_back({"i_o","j","k_o","i_i","k_i"});
            } else if (ti == 1 && tj > 1 && tk > 1) {
                orders.push_back({"i","j_o","k_o","j_i","k_i"});
            } else {
                // ti==1, tj==1, tk==1
                orders.push_back({"i","j","k"});
            }
            return orders;
        };

        AutoTileReport report;
        report.M = M;
        report.K = K;
        report.N = N;
        report.best.time_ms = 1e18;

        for (int ti : tile_ij_cands) {
            if (M % ti != 0 || ti > M) continue;
            for (int tj : tile_ij_cands) {
                if (N % tj != 0 || tj > N) continue;
                for (int tk : tile_k_cands) {
                    if (K % tk != 0 || tk > K) continue;

                    auto orders = make_orders(ti, tj, tk);
                    for (auto& order : orders) {
                        std::string kernel = codegen_with_config(
                            M, K, N, ti, tj, tk, order, false,
                            lhs_->name(), rhs_->name());

                        std::string prog = generate_benchmark_program(kernel, M, K, N, benchmark_runs, ti, tj, tk);

                        std::string tag = std::to_string(ti) + "_" + std::to_string(tj) + "_" + std::to_string(tk);
                        for (auto& o : order) tag += "_" + o;
                        std::string src_path = "/tmp/namiloop_bench_" + tag + ".cpp";
                        std::string bin_path = "/tmp/namiloop_bench_" + tag;
                        {
                            std::ofstream f(src_path);
                            f << prog;
                        }

                        std::string compile_cmd = "g++ -std=c++17 -O2 -o " + bin_path + " " + src_path + " 2>/dev/null";
                        int ret = std::system(compile_cmd.c_str());
                        if (ret != 0) {
                            std::remove(src_path.c_str());
                            continue;
                        }

                        FILE* pipe = popen(bin_path.c_str(), "r");
                        if (!pipe) {
                            std::remove(src_path.c_str());
                            std::remove(bin_path.c_str());
                            continue;
                        }

                        char buf[256];
                        double time_ms = 1e18;
                        if (fgets(buf, sizeof(buf), pipe)) {
                            time_ms = std::atof(buf);
                        }
                        pclose(pipe);

                        std::remove(src_path.c_str());
                        std::remove(bin_path.c_str());

                        TileResult tr;
                        tr.tile_i = ti;
                        tr.tile_j = tj;
                        tr.tile_k = tk;
                        tr.loop_order = order;
                        tr.time_ms = time_ms;
                        report.results.push_back(tr);

                        if (time_ms < report.best.time_ms) {
                            report.best = tr;
                            report.kernel_code = kernel;
                        }
                    }
                }
            }
        }

        std::sort(report.results.begin(), report.results.end(),
                  [](const TileResult& a, const TileResult& b) {
                      return a.time_ms < b.time_ms;
                  });

        return report;
    }

private:
    void add_loop(const std::string& name, int lo, int hi) {
        LoopVar lv(name, lo, hi);
        lv.id_ = next_id_++;
        lv.owner_ = this;
        loops_.push_back(lv);
    }

    int find_loop(const LoopVar& var) const {
        // 先用 id 找
        for (size_t i = 0; i < loops_.size(); i++) {
            if (loops_[i].id_ == var.id_) return (int)i;
        }
        // 再用 name 找
        for (size_t i = 0; i < loops_.size(); i++) {
            if (loops_[i].name_ == var.name_) return (int)i;
        }
        return -1;
    }

    static bool is_var_ready(const std::string& base, int tile,
                             const std::vector<std::string>& order, int before_pos) {
        if (tile <= 1) {
            // 不需要 split，只要 base 在 before_pos 之前
            for (int i = 0; i < before_pos; i++) {
                if (order[i] == base) return true;
            }
            return false;
        }
        // 需要 outer 和 inner 都在 before_pos 之前
        bool has_o = false, has_i = false;
        for (int i = 0; i < before_pos; i++) {
            if (order[i] == base + "_o") has_o = true;
            if (order[i] == base + "_i") has_i = true;
        }
        return has_o && has_i;
    }

    // v2 codegen 用 loops_ 的當前順序
    std::string codegen_v2(const std::string& header, int M, int K, int N) const {
        // 收集 tile info
        int ti = 1, tj = 1, tk = 1;
        for (auto& s : splits_) {
            if (s.orig == "i") ti = s.factor;
            else if (s.orig == "j") tj = s.factor;
            else if (s.orig == "k") tk = s.factor;
        }

        // 建立 loop order
        std::vector<std::string> order;
        for (auto& l : loops_) order.push_back(l.name_);

        // 找第一個 parallel
        bool do_parallel = false;
        for (auto& l : loops_) {
            if (l.is_parallel_) { do_parallel = true; break; }
        }

        return codegen_with_config(M, K, N, ti, tj, tk, order, do_parallel,
                                   lhs_->name(), rhs_->name());
    }

    std::string generate_benchmark_program(const std::string& kernel,
                                           int M, int K, int N,
                                           int runs,
                                           int ti = 1, int tj = 1, int tk = 1) const {
        std::ostringstream out;
        out << "#include <chrono>\n";
        out << "#include <iostream>\n";
        out << "#include <vector>\n";
        out << "#include <algorithm>\n";
        out << "#include <cstdlib>\n\n";
        out << "int main() {\n";
        out << "    const int M = " << M << ", K = " << K << ", N = " << N << ";\n";
        out << "    std::vector<double> A(M * K), B(K * N), C(M * N);\n";
        out << "    for (int i = 0; i < M * K; i++) A[i] = (double)(rand() % 100) / 100.0;\n";
        out << "    for (int i = 0; i < K * N; i++) B[i] = (double)(rand() % 100) / 100.0;\n";
        out << "    double* A_ptr = A.data();\n";
        out << "    double* B_ptr = B.data();\n";
        out << "    double* C_ptr = C.data();\n\n";
        out << "    std::vector<double> times;\n";
        out << "    for (int run = 0; run < " << runs << "; run++) {\n";
        out << "        // 初始化 C = 0\n";
        out << "        for (int i = 0; i < M * N; i++) C[i] = 0;\n";
        out << "        auto t0 = std::chrono::high_resolution_clock::now();\n";
        out << "        {\n";
        out << "            double* A = A_ptr;\n";
        out << "            double* B = B_ptr;\n";
        out << "            double* C = C_ptr;\n";
        out << kernel;
        out << "        }\n";
        out << "        auto t1 = std::chrono::high_resolution_clock::now();\n";
        out << "        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();\n";
        out << "        times.push_back(ms);\n";
        out << "    }\n";
        out << "    std::sort(times.begin(), times.end());\n";
        out << "    double median = times[times.size() / 2];\n";
        out << "    std::cout << median << std::endl;\n";
        out << "    return 0;\n";
        out << "}\n";
        return out.str();
    }
};

// ============================================================================
// LoopVar::split — 實作（需要 Expr 完整定義）
// ============================================================================
inline std::pair<LoopVar, LoopVar> LoopVar::split(int factor) {
    assert(owner_ && "LoopVar must belong to an Expr");
    return owner_->split_var(*this, factor);
}

// ============================================================================
// operator* — 建立 matmul AST
// ============================================================================
inline Expr operator*(const Tensor& lhs, const Tensor& rhs) {
    assert(lhs.cols() == rhs.rows() && "矩陣維度不匹配！");
    return Expr(Expr::MATMUL, lhs, rhs);
}

} // namespace namiloop
