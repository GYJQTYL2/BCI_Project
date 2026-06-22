"""
通道质量评估

在基线校正之前对原始 ADC 值进行评估，识别三种硬件失效模式：
  1. ADC 饱和（接触过紧/过多导电膏）：大量样本在 0 或 1450
  2. 电极脱落（断路）：信号近似直线，std 极低
  3. 高频噪声（高阻抗/EMG 混入）：30Hz 以上功率占比过高

返回 "good" / "poor" / "bad" 三级质量，供下游动态选通道。
"""

from __future__ import annotations

import numpy as np

# Muse ADC 范围 [0, 1450]，边界值即饱和
_SAT_LOW = 1.0
_SAT_HIGH = 1449.0

# 静态阈值
_FLATLINE_STD = 1.0      # std < 1µV → 脱落
_SAT_BAD = 0.10          # 饱和率 > 10% → bad
_SAT_POOR = 0.05         # 饱和率 > 5%  → poor（可插值）
_HF_BAD = 0.60           # 高频功率占比 > 60% → bad
_HF_POOR = 0.40          # 高频功率占比 > 40% → poor


def assess(signal: np.ndarray, fs: float = 256.0) -> dict:
    """
    评估单通道单窗口的信号质量。

    参数:
        signal : 原始 ADC 值数组（未做任何预处理）
        fs     : 采样率（Hz）

    返回:
        {
          "quality"  : "good" | "poor" | "bad",
          "sat_rate" : float,   # 饱和样本比例 [0, 1]
          "flatline" : bool,    # 是否脱落
          "hf_ratio" : float,   # 高频功率占比 [0, 1]
          "reason"   : str,     # 降级原因，quality=="good" 时为 ""
        }
    """
    signal = np.asarray(signal, dtype=float)

    # 1. 饱和率
    sat_rate = float(np.mean((signal <= _SAT_LOW) | (signal >= _SAT_HIGH)))

    # 2. 平线检测
    flatline = bool(signal.std() < _FLATLINE_STD)

    # 3. 高频噪声比（需要在去 DC 后计算）
    hf_ratio = _hf_ratio(signal, fs)

    # 综合判定
    quality, reason = _judge(sat_rate, flatline, hf_ratio)

    return {
        "quality":  quality,
        "sat_rate": sat_rate,
        "flatline": flatline,
        "hf_ratio": hf_ratio,
        "reason":   reason,
    }


def assess_window(raw_df, channels: list[str], fs: float = 256.0) -> dict[str, dict]:
    """
    对一个窗口 DataFrame 的多个通道批量评估。

    参数:
        raw_df   : 含原始 ADC 值的 DataFrame（基线校正之前）
        channels : 待评估通道列表，如 ["CH1","CH2","CH3","CH4"]
        fs       : 采样率

    返回:
        {通道名: assess() 结果} 的字典
    """
    return {ch: assess(raw_df[ch].values, fs) for ch in channels if ch in raw_df.columns}


def good_channels(quality_map: dict[str, dict]) -> list[str]:
    """从 assess_window() 结果中提取 good/poor 通道列表（排除 bad）"""
    return [ch for ch, q in quality_map.items() if q["quality"] != "bad"]


def interpolate_saturated(signal: np.ndarray) -> np.ndarray:
    """
    对 poor 通道的零散饱和点做线性插值。
    只处理饱和率 < _SAT_BAD 的情况，调用前应先确认 quality == "poor"。
    """
    signal = signal.copy().astype(float)
    sat_mask = (signal <= _SAT_LOW) | (signal >= _SAT_HIGH)
    if not sat_mask.any():
        return signal

    indices = np.arange(len(signal))
    good = ~sat_mask
    if good.sum() < 2:
        return signal  # 好点太少，无法插值

    signal[sat_mask] = np.interp(indices[sat_mask], indices[good], signal[good])
    return signal


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _hf_ratio(signal: np.ndarray, fs: float) -> float:
    """计算 >30Hz 功率占 1–45Hz 总功率的比例"""
    n = len(signal)
    if n < 32:
        return 0.0
    try:
        from scipy.signal import welch
        nperseg = min(256, n)
        freqs, psd = welch(signal - signal.mean(), fs=fs, nperseg=nperseg)
        mask_total = (freqs >= 1.0) & (freqs <= 45.0)
        mask_hf    = (freqs >= 30.0) & (freqs <= 45.0)
        total = psd[mask_total].sum()
        if total < 1e-12:
            return 0.0
        return float(psd[mask_hf].sum() / total)
    except Exception:
        return 0.0


def _judge(sat_rate: float, flatline: bool, hf_ratio: float):
    if flatline:
        return "bad", "flatline"
    if sat_rate > _SAT_BAD:
        return "bad", f"sat={sat_rate:.1%}"
    if hf_ratio > _HF_BAD:
        return "bad", f"hf={hf_ratio:.1%}"
    if sat_rate > _SAT_POOR:
        return "poor", f"sat={sat_rate:.1%}"
    if hf_ratio > _HF_POOR:
        return "poor", f"hf={hf_ratio:.1%}"
    return "good", ""
