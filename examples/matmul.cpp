// ============================================================================
// NamiLoop 範例：自動搜索最佳 tile size + reorder 組合
// ============================================================================
#include "namiloop/namiloop.hpp"
using namespace namiloop;

int main() {
    Tensor A("A", 64, 32);
    Tensor B("B", 32, 64);
    auto C = A * B;

    // 自動搜索最佳 tile + reorder
    auto result = C.auto_tile();
    result.print_report();
    result.save("best_kernel.inc");

    return 0;
}
