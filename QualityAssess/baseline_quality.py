"""
ASR 基线质量评估

在 finish_recording() 训练完成后调用，检查基线是否可用。
不合格时返回失败原因，由调用方决定是否丢弃基线。

检查项：
  1. 基线时长        < 30s → warn；< 15s → fail
  2. 通道功率均衡性  用 cov 对角线（通道方差）检测坏道，任一通道 > 其他均值的 N 倍 → fail
                     同时检测 NaN/Inf，任意两通道同时劣化也能被捕捉（比 M 对角更可靠）
  3. 协方差可用性    cov is None → fail（fit 未获得足够干净窗口）
  4. T 矩阵有效性    T 含 NaN/Inf → fail（阈值计算崩溃）；T 最大绝对值 > 阈值 → warn
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

# 阈值
_MIN_SECONDS_FAIL = 15.0     # 低于此时长直接失败
_MIN_SECONDS_WARN = 30.0     # 低于此时长发出警告
_POWER_IMBALANCE  = 3.0      # 某通道方差 > 其他通道均值的 N 倍 → 坏道污染
_T_EXTREME        = 50.0     # T 矩阵元素绝对值超过此值 → 阈值可能被噪声扭曲（warn）


def assess_asr_baseline(asr, n_samples: int | None, sfreq: float) -> dict:
    """
    评估 ASR 对象的基线质量。

    参数:
        asr       : 训练好的 asrpy.ASR 对象
        n_samples : 基线样本总数；None 表示未知（加载时跳过时长检查）
        sfreq     : 采样率（Hz）

    返回:
        {
          "ok"      : bool,        # True = 可用，False = 不建议使用
          "warns"   : list[str],   # 警告列表（ok=True 时也可能有）
          "errors"  : list[str],   # 失败原因列表（ok=False）
        }
    """
    warns = []
    errors = []

    # 1. 时长检查（仅 n_samples 已知时）
    if n_samples is not None:
        duration = n_samples / sfreq
        if duration < _MIN_SECONDS_FAIL:
            errors.append(f"基线时长 {duration:.1f}s 过短（最低要求 {_MIN_SECONDS_FAIL}s）")
        elif duration < _MIN_SECONDS_WARN:
            warns.append(f"基线时长 {duration:.1f}s，建议 >= {_MIN_SECONDS_WARN}s")
    else:
        duration = None

    # 2. 协方差可用性（必须先于通道功率检查）
    if hasattr(asr, 'cov') and asr.cov is None:
        errors.append("ASR cov=None：fit 未获得足够干净窗口，transform 行为不可预测")

    # 3. 通道功率均衡性（用 cov 对角线 = 各通道方差，直接反映功率）
    #    M 对角线是白化矩阵系数，不直接等于通道功率，容易误判
    if hasattr(asr, 'cov') and asr.cov is not None:
        cov_diag = np.diag(asr.cov)
        # 先检查 NaN/Inf
        if not np.all(np.isfinite(cov_diag)):
            errors.append("ASR cov 对角线含 NaN/Inf，基线协方差矩阵损坏")
        else:
            for i, val in enumerate(cov_diag):
                others = np.delete(cov_diag, i)
                others_mean = others.mean()
                if others_mean > 0 and val > others_mean * _POWER_IMBALANCE:
                    errors.append(
                        f"CH{i+1} 基线方差 {val:.1f} 是其他通道均值 {others_mean:.1f} 的 "
                        f"{val/others_mean:.1f} 倍，录制时该通道可能在饱和或有大伪迹"
                    )

    # 4. T 矩阵有效性
    if hasattr(asr, 'T') and asr.T is not None:
        if not np.all(np.isfinite(asr.T)):
            errors.append("T 矩阵含 NaN/Inf，ASR 阈值计算崩溃，不可用")
        else:
            t_max = np.abs(asr.T).max()
            if t_max > _T_EXTREME:
                warns.append(
                    f"T 矩阵最大绝对值 {t_max:.1f}（>{_T_EXTREME}），"
                    "伪迹阈值可能被噪声扭曲，建议重录基线"
                )

    ok = len(errors) == 0
    dur_str = f"{duration:.1f}s" if duration is not None else "时长未知"

    # 输出 log
    if ok and not warns:
        log.info(f"[ASR基线] 质量良好：{dur_str}，通道功率均衡")
    elif ok:
        log.warning(f"[ASR基线] 可用但有警告（{dur_str}）：" + "；".join(warns))
    else:
        log.error(f"[ASR基线] 不合格，将禁用 ASR：" + "；".join(errors))
        if warns:
            log.warning(f"[ASR基线] 附加警告：" + "；".join(warns))

    return {"ok": ok, "warns": warns, "errors": errors}
