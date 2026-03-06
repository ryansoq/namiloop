# 🌊 NamiLoop — 自動化迴圈優化 DSL

> **演算法與排程分離** — 用漂亮的 C++17 語法糖定義矩陣運算，Halide/Tiramisu 風格細粒度排程，自動窮舉搜索最佳 tile size + reorder 組合。

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

// 取得迴圈變數（自動從 matmul 推導出 i, j, k）
auto [i, j, k] = C.loops();

// Halide/Tiramisu 風格細粒度排程
auto [i_o, i_i] = i->split(8);   // i → i_o, i_i
auto [j_o, j_i] = j->split(4);   // j → j_o, j_i
auto [k_o, k_i] = k->split(16);  // k → k_o, k_i

// 指定完整迴圈順序
C.reorder({i_o, j_o, k_o, i_i, j_i, k_i});

i_o->parallel();     // 最外層並行化
k_i->vectorize();    // 最內層向量化

C.codegen("kernel.inc");

// 或自動搜索最佳配置
auto best = C.auto_tile();
best.print_report();
best.save("best_kernel.inc");
```

---

## 排程 API

| API | 說明 |
|-----|------|
| `C.loops()` | 取得 `(i, j, k)` 迴圈變數 |
| `i->split(factor)` | 分割迴圈 → 回傳 `(i_o, i_i)` |
| `C.swap(a, b)` | 交換任意兩層迴圈 |
| `C.reorder({...})` | 指定完整迴圈順序 |
| `C.tile(i, j, fi, fj)` | 組合技 = split + split + reorder |
| `i->parallel()` | 加 `#pragma omp parallel for` |
| `i->vectorize()` | 加 `#pragma omp simd` |

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
              │  swap / reorder │
              │  parallel       │
              │  vectorize      │
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
  └──────────────┘         │  tile + reorder │
                           │  編譯+測時    │
                           └──────────────┘
```

---

## 執行流程

```
DSL 定義 → AST → split/reorder → CodeGen (ax+b) → Benchmark → 🏆 最佳 Kernel
```

---

## Loop 變換

### split(factor)
```
for i in [0, N)  →  for i_o in [0, N/f)
                        for i_i in [0, f)
                            i = f * i_o + i_i   // 仿射索引 ax+b
```

### swap(var1, var2)
交換任意兩層迴圈。

### reorder({...})
指定完整的迴圈順序 — 六層 tiled matmul：
```
原始: for i → for j → for k
tile 後: for i_o → for j_o → for k_o → for i_i → for j_i → for k_i
```

### tile(dim_i, dim_j, fi, fj)
組合技：`split(i, fi) + split(j, fj) + reorder`

### parallel() / vectorize()
```cpp
i_o->parallel();   // #pragma omp parallel for
k_i->vectorize();  // #pragma omp simd
```

---

## Auto-Tile 搜索

自動嘗試不同的 tile size × reorder 組合：

```
🔍 Auto-Tile Search: A(64,32) × B(32,64)
  tile(8,4,4) ikj              → 0.08ms  ← 🏆 Best!
  tile(4,8,8) ijk              → 0.09ms
  tile(16,16,4) kij            → 0.12ms
  ...
🏆 Best config: tile(8,4,4) ikj → 0.08ms
📄 Saved to: best_kernel.inc
```

搜索空間：
- tile_i, tile_j, tile_k ∈ {1, 2, 4, 8, 16, 32}
- reorder ∈ {ijk, ikj, jik, kij}
- 過濾不合法組合 → 編譯 → benchmark 取中位數 → 選最快

---

## 快速開始

```bash
make test            # 12 項測試
make example_manual  # 手動排程：六層 tiled matmul
make example_matmul  # Auto-Tile 搜索（需要幾分鐘）
```

---

## 設計哲學

| 原則 | 說明 |
|------|------|
| **演算法/排程分離** | 改排程不需改演算法 |
| **Expression Template** | `A * B` 不執行，只建構 AST |
| **LoopVar 控制代碼** | split 回傳新 var，可繼續排程 |
| **仿射索引 (ax+b)** | 致敬 Ryan 的 autoloop |
| **Header-Only** | 單一 `.hpp`，零依賴 |
| **自動探索** | 窮舉 tile × reorder，讓數據說話 |

---

## 專案結構

```
namiloop/
├── include/namiloop/namiloop.hpp   # header-only 主檔
├── examples/
│   ├── matmul.cpp                  # auto-tile demo
│   └── manual.cpp                  # 手動排程 demo
├── tests/test_namiloop.cpp         # 12 項測試
├── Makefile
├── README.md
└── LICENSE (MIT)
```

---

*NamiLoop — 讓迴圈自己找到最快的路 🌊*
