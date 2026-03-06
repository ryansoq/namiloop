// ============================================================================
// NamiLoop 範例：手動排程
// ============================================================================
#include "namiloop/namiloop.hpp"
using namespace namiloop;

int main() {
    Tensor A("A", 16, 8);
    Tensor B("B", 8, 16);
    auto C = A * B;  // 只記錄 AST，不計算

    // 手動排程
    C.tile(4, 4);
    C.parallel();

    // 印出生成的程式碼
    std::string code = C.codegen();
    std::cout << "=== 生成的 Kernel ===\n";
    std::cout << code << std::endl;

    // 寫入檔案
    C.codegen("manual_kernel.inc");
    std::cout << "📄 已儲存至 manual_kernel.inc\n";

    return 0;
}
