# tenghe3 — 山东电力市场电价预测

15 分钟分辨率数据上的 **日前（DA）电价** LightGBM 基准、**实时（RT）电价** PatchTST 类多步预测（6 小时 / 24 步）、**外生变量** 6 小时预测，以及 Excel 审计与外生预测精度分析。

默认数据：`山东-全年-带时间点.xlsx`（自备，或通过各脚本 `--data_path` 指定）。

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
