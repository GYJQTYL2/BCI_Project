"""
EEG 认知负荷指数计算

基于频域特征 DataFrame（来自 EEGFeatureExtractor）计算认知负荷指标:

    cognitive_load_index : theta / alpha  (升高表示认知负荷增加)
    cog_load_score       : 归一化负荷得分 [0, 1]，值越高表示认知负荷越重

EEG 依据:
    - Theta (4-8 Hz) 随工作记忆负荷增加而升高（前额叶 Theta 同步）
    - Alpha (8-13 Hz) 随认知负荷增加而降低（皮层激活导致 Alpha 去同步化）
    - Theta/Alpha 比值是文献中最常用的认知负荷 EEG 指标

参考: Klimesch (1999), Gevins & Smith (2003)
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


class CognitiveLoadIndex:
    """
    EEG 认知负荷指数计算器

    参数:
        channels : 参与计算的通道列表，默认 CH1-CH4
    """

    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or _CHANNELS

    def cognitive_load_index(self, row: pd.Series) -> float:
        """
        认知负荷指数: θ / α

        升高时表示认知负荷增加（工作记忆压力大）；
        降低时表示任务轻松或放松状态。
        """
        theta = _band_mean(row, "theta", self.channels)
        alpha = _band_mean(row, "alpha", self.channels)
        return theta / (alpha + _EPS)

    def cog_load_score(self, row: pd.Series) -> float:
        """
        归一化认知负荷得分 [0, 1]

        基于 Theta/Alpha 比值的 sigmoid 压缩：
            CI = 0.5 → score ≈ 0.18  (低负荷)
            CI = 1.5 → score ≈ 0.50  (中等负荷)
            CI = 3.0 → score ≈ 0.88  (高负荷)
        """
        ci = self.cognitive_load_index(row)
        return float(1.0 / (1.0 + np.exp(-2.0 * (ci - 1.5))))

    def compute(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        对特征 DataFrame 的每行计算认知负荷指标

        参数:
            df_features : EEGFeatureExtractor 输出的特征 DataFrame
        返回:
            DataFrame，列为: timestamp（若有）| cognitive_load_index | cog_load_score
        """
        if df_features.empty:
            return pd.DataFrame()

        records = []
        for _, row in df_features.iterrows():
            rec: dict = {}
            if "timestamp" in df_features.columns:
                rec["timestamp"] = row["timestamp"]
            rec["cognitive_load_index"] = self.cognitive_load_index(row)
            rec["cog_load_score"] = self.cog_load_score(row)
            records.append(rec)

        return pd.DataFrame(records)
