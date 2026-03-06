// ============================================================================
// NamiLoop 範例：Halide/Tiramisu 風格手動排程
// ============================================================================
#include "namiloop/namiloop.hpp"
using namespace namiloop;

int main() {
    Tensor A("A", 32, 16);
    Tensor B("B", 16, 32);
    auto C = A * B;  // 只記錄 AST，不計算

    // 取得迴圈變數
    auto [i, j, k] = C.loops();

    // 細粒度排程 — 跟 Halide 一樣！
    auto [i_o, i_i] = i->split(8);   // i → i_o(4), i_i(8)
    auto [j_o, j_i] = j->split(8);   // j → j_o(4), j_i(8)
    auto [k_o, k_i] = k->split(4);   // k → k_o(4), k_i(4)

    // 重排迴圈順序：六層 tiled matmul
    C.reorder({i_o, j_o, k_o, i_i, j_i, k_i});

    // 最外層並行化
    i_o->parallel();

    // 最內層向量化
    k_i->vectorize();

    // 印出生成的程式碼
    std::string code = C.codegen();
    std::cout << "=== 六層 Tiled Matmul Kernel ===\n";
    std::cout << code << std::endl;

    // 寫入檔案
    C.codegen("manual_kernel.inc");
    std::cout << "📄 已儲存至 manual_kernel.inc\n";

    return 0;
}
