# 🌊 NamiLoop — 自動化迴圈優化 DSL

> **演算法與排程分離** — 用漂亮的 C++17 語法糖定義矩陣運算，自動窮舉搜索最佳 tile size。

**Authors:** Ryan & Nami  
**License:** MIT  
**語言:** C++17, Header-Only, 零外部依賴

---

## 設計理念

受 [Halide](https://halide-lang.org/)、[Tiramisu](https://tiramisu-compiler.org/) 以及 Ryan 的 autoloop 啟發，NamiLoop 將**演算法定義**與**執行排程**完全分離：

- 🎯 使用者用直覺的語法描述「算什麼」
- ⚡ 框架負責「怎麼算最快」

---

## 語法糖範例

```cpp
#include "namiloop/namiloop.hpp"
using namespace namiloop;

Tensor A("A", 100, 50);
Tensor B("B", 50, 100);
auto C = A * B;         // 不執行，只記錄運算 AST

// 手動排程
C.tile(4, 4);
C.parallel();
C.codegen("kernel.inc");

// 或自動搜索
auto best = C.auto_tile();  // 窮舉 → 編譯 → benchmark → 選最佳
best.print_report();
best.save("best_kernel.inc");
```

---

## 框架架構

```
┌─────────────────────────────────────────────────┐
│                  使用者程式碼                      │
│         Tensor A * B  →  Expr (AST)              │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │   排程引擎       │
              │  split / tile   │
              │  reorder        │
              │  parallel       │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   CodeGen       │
              │  AST → C 迴圈   │
              │  仿射索引 ax+b  │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼                         ▼
  ┌──────────────┐         ┌──────────────┐
  │  手動輸出     │         │  Auto-Tile   │
  │  kernel.inc  │         │  窮舉搜索     │
  └──────────────┘         │  編譯+測時    │
                           │  選最佳       │
                           └──────────────┘
```

---

## 執行流程

```
DSL 定義 → AST 建立 → Loop 變換 → CodeGen → Benchmark → 🏆 最佳 Kernel
```

1. `Tensor A * B` 建立 `Expr` AST 節點
2. `.tile()` / `.split()` / `.parallel()` 記錄排程指令
3. `.codegen()` 走訪 AST，生成帶仿射索引的 C 迴圈
4. `auto_tile()` 窮舉所有候選 tile size → 編譯 → benchmark → 選最快的

---

## Loop 變換

### split(dim, factor)
把一層迴圈分成外層 + 內層：
```
for i in [0, N)  →  for i_o in [0, N/f)
                        for i_i in [0, f)
                            i = f * i_o + i_i   // 仿射索引
```

### reorder(dim1, dim2)
交換兩層迴圈的執行順序。

### tile(ti, tj)
組合技：`split(i, ti)` + `split(j, tj)` + `reorder`，產生四層迴圈。

### parallel(dim)
加上 `#pragma omp parallel for`，啟用 OpenMP 並行。

---

## Auto-Tile 搜索流程

```
🔍 Auto-Tile Search: A(64,32) × B(32,64)
  tile(8,8)    → 0.2ms  ← 🏆 Best!
  tile(4,4)    → 0.3ms
  tile(16,16)  → 0.3ms
  ...
🏆 Best config: tile(8,8) → 0.2ms
📄 Saved to: best_kernel.inc
```

搜索策略：
1. 列舉候選 tile sizes（{1, 2, 4, 8, 16, 32}）
2. 過濾不合法的（維度不能整除的跳過）
3. 對每個候選：codegen → g++ -O2 編譯 → 多次執行取中位數
4. 排序，選最快的
5. 輸出最佳 kernel + 報告

---

## 快速開始

```bash
# 編譯並執行測試
make test

# Auto-Tile 範例
make example_matmul

# 手動排程範例
make example_manual
```

---

## 設計哲學

| 原則 | 說明 |
|------|------|
| **演算法/排程分離** | 改排程不需改演算法，改演算法不需改排程 |
| **Expression Template** | `A * B` 不執行運算，只建構 AST |
| **仿射索引 (ax+b)** | 致敬 Ryan 的 autoloop，所有迴圈索引用仿射表達式 |
| **Header-Only** | 單一 `.hpp`，零設定零依賴 |
| **自動探索** | 不猜、不假設 — 窮舉搜索，讓數據說話 |

---

## 專案結構

```
namiloop/
├── include/namiloop/namiloop.hpp   # header-only 主檔
├── examples/
│   ├── matmul.cpp                  # auto-tile demo
│   └── manual.cpp                  # 手動排程 demo
├── tests/test_namiloop.cpp         # 測試
├── Makefile
├── README.md
└── LICENSE (MIT)
```

---

*NamiLoop — 讓迴圈自己找到最快的路 🌊*
