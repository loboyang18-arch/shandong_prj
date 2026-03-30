# tenghe3 — 山东电力市场电价预测

15 分钟分辨率数据上的 **日前（DA）电价** LightGBM 基准、**实时（RT）电价** PatchTST 类多步预测（6 小时 / 24 步）、**外生变量** 6 小时预测，以及 Excel 审计与外生预测精度分析。

默认数据：`山东-全年-带时间点.xlsx`（已纳入本仓库；亦可通过各脚本 `--data_path` 指定其他路径）。

---

## 仓库结构

| 脚本 | 说明 |
|------|------|
| `da_price_baseline.py` | 日前基准入口，调用 `da_price_baseline_v3_segmented.py` |
| `da_price_baseline_v3_segmented.py` | 分三段 LightGBM、D-1 09:00 约束 |
| `rt_patchtst_onefile.py` | RT 单文件管线（内训外生 + PatchTST），支持 `fit` / `predict` |
| `exog_pipeline_rt6h_v2.py` | 外生 6h（24 步）预测管线 |
| `analyze_exog_forecast_accuracy.py` | 对比 Excel 中「预测值」与「实际值」列的精度 |
| `audit_shandong_excel.py` | 时间轴、覆盖率、列清单等审计 |

运行产物默认写入 `outputs_da_price/`、`outputs_rt_patchtst/`、`outputs_exog_rt6h/`、`outputs_exog_accuracy/`（已 `.gitignore`，需自行跑脚本生成）。

---

## 环境

Python 3.9+，安装依赖：

```bash
pip install -r requirements.txt
```

GPU 训练请从 [PyTorch 官网](https://pytorch.org/) 安装与 CUDA 匹配的 `torch`。

### macOS：LightGBM 报 `libomp.dylib` 找不到

PyPI 上的 **macOS 预编译 wheel** 会动态链接 OpenMP。任选其一：

1. **安装 OpenMP 运行时（推荐，多线程性能更好）**  
   先更新 Homebrew（过旧的 core 可能没有新版 bottle），再安装：
   ```bash
   brew update && brew install libomp
   ```
2. **不装 libomp：从源码编译并关闭 OpenMP**（训练可能略慢，但不依赖系统 libomp；本仓库已在 arm64 + macOS 14 上验证可行）：
   ```bash
   pip install cmake ninja
   pip uninstall -y lightgbm
   CMAKE_ARGS="-DUSE_OPENMP=OFF" pip install --no-cache-dir --no-binary lightgbm "lightgbm>=4.0,<5"
   ```
   若已按 `requirements.txt` 装过 wheel，需先执行上面的 `uninstall` 再重装。

---

## 快速命令

```bash
# 日前
python da_price_baseline.py --data_path "山东-全年-带时间点.xlsx" --outdir outputs_da_price --run_id da_baseline

# 实时（深度学习）
python rt_patchtst_onefile.py --data_path "山东-全年-带时间点.xlsx" --outdir outputs_rt_patchtst --run_id rt_onefile

# 外生 6h
python exog_pipeline_rt6h_v2.py --help

# 外生预测列精度
python analyze_exog_forecast_accuracy.py --help

# 数据审计
python audit_shandong_excel.py --data_path "山东-全年-带时间点.xlsx" --outdir audit_outputs
```

---

## 说明

- 日前基准强调 **D-1 日 09:00** 前可得信息；外生特征优先使用预测列，缺失时回退实际列存在评估偏乐观风险，见 `da_price_baseline_v3_segmented.py` 注释。
- RT 管线训练不使用 Excel 中带「预测值」的列（设计见 `rt_patchtst_onefile.py` 文件头）。
