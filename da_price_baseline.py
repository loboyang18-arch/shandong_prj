# -*- coding: utf-8 -*-
"""
日前电价预测基准脚本（当前默认基准）
===================================

本工程当前默认的日前电价预测基准实现为：
`da_price_baseline_v3_segmented.py`

该实现基于“三段分时段分模型 + 周周期特征 + D-1 同时刻特征 + 晚高峰高价样本加权”。

使用方法示例：
  python da_price_baseline.py --data_path "山东-全年-带时间点.xlsx" --outdir outputs_da_price --run_id da_baseline
"""

from da_price_baseline_v3_segmented import main


if __name__ == "__main__":
    main()

