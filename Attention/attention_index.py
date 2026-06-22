"""
EEG 注意力指数计算

基于频域特征 DataFrame（来自 EEGFeatureExtractor）计算三种注意力指标:

    theta_alpha_ratio  : theta / alpha  (升高表示疲劳/走神)
    engagement_index   : beta / (alpha + theta)  (升高表示主动投入)
    attention_score    : 归一化综合得分 [0, 1]，值越高表示注意力越集中

所有方法均支持多通道平均，以 CH1-CH4 的均值作为全局指标。

典型特征列名格式（来自 frequency_domain.py）:
    CH1_theta_power, CH1_alpha_power, CH1_beta_power, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
_EPS = 1e-10


def _band_mean(row: pd.Series, band: str, channels: list[str]) -> float:
    """取指定频段在多通道上的均值功率，跳过 NaN（坏道）"""
    vals = [row.get(f"{ch}_{band}_power", np.nan) for ch in channels]
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else 0.0


class AttentionIndex:
    """
    EEG 注意力指数计算器

    参数:
        channels : 参与计算的通道列表，默认 CH1-CH4
    """

    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or _CHANNELS

    # ── 单行指标 ──────────────────────────────────────────────────────────

    def theta_alpha_ratio(self, row: pd.Series) -> float:
        """
        θ/α 比值

        升高时表示大脑处于放松/疲劳/走神状态；下降时注意力更集中。
        """
        theta = _band_mean(row, "theta", self.channels)
        alpha = _band_mean(row, "alpha", self.channels)
        return theta / (alpha + _EPS)

    def engagement_index(self, row: pd.Series) -> float:
        """
        参与度指数: β / (α + θ)

        Pope et al. (1995) 提出；值越高表示认知负荷与主动投入越强。
        """
        beta  = _band_mean(row, "beta",  self.channels)
        alpha = _band_mean(row, "alpha", self.channels)
        theta = _band_mean(row, "theta", self.channels)
        return beta / (alpha + theta + _EPS)

    def attention_score(self, row: pd.Series) -> float:
        """
        归一化注意力得分 [0, 1]

        基于参与度指数的 sigmoid 压缩。值越接近 1 表示注意力越集中。
        """
        ei = self.engagement_index(row)
        # sigmoid 以 ei=1.0 为中点，斜率 k=3 适配典型 EEG 范围
        return float(1.0 / (1.0 + np.exp(-3.0 * (ei - 1.0))))

    # ── DataFrame 批量计算 ────────────────────────────────────────────────

    def compute(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        对特征 DataFrame 的每一行计算注意力指标

        参数:
            df_features : EEGFeatureExtractor 输出的特征 DataFrame
        返回:
            DataFrame，列为:
                timestamp (若原始数据有) | theta_alpha_ratio |
                engagement_index | attention_score
        """
        if df_features.empty:
            return pd.DataFrame()

        records = []
        for _, row in df_features.iterrows():
            rec: dict = {}
            if "timestamp" in df_features.columns:
                rec["timestamp"] = row["timestamp"]
            rec["theta_alpha_ratio"] = self.theta_alpha_ratio(row)
            rec["engagement_index"]  = self.engagement_index(row)
            rec["attention_score"]   = self.attention_score(row)
            records.append(rec)

        return pd.DataFrame(records)
