"""
时域特征提取

每个通道提取以下特征:
    均值 / 标准差 / 方差 / 偏度 / 峰度
    峰峰值 / 过零率
    Hjorth 参数 (活动性 / 移动性 / 复杂性)
"""

import numpy as np
from scipy import stats


def extract(epoch: np.ndarray, ch_name: str) -> dict:
    """
    提取单通道单 epoch 的时域特征

    参数:
        epoch   : 一维 numpy 数组，长度为 epoch 样本数
        ch_name : 通道名，用于构造特征列名前缀
    返回:
        {特征名: 特征值} 的字典
    """
    feat = {}
    p = ch_name  # prefix

    std = epoch.std()

    feat[f"{p}_mean"]     = epoch.mean()
    feat[f"{p}_std"]      = std
    feat[f"{p}_var"]      = epoch.var()
    feat[f"{p}_skew"]     = stats.skew(epoch)
    feat[f"{p}_kurtosis"] = stats.kurtosis(epoch)
    feat[f"{p}_ptp"]      = epoch.max() - epoch.min()

    # 过零率
    feat[f"{p}_zcr"] = float(np.sum(np.diff(np.sign(epoch)) != 0)) / len(epoch)

    # Hjorth 参数
    d1 = np.diff(epoch)
    d2 = np.diff(d1)
    std_d1 = d1.std()
    std_d2 = d2.std()

    activity   = float(std ** 2)
    mobility   = float(std_d1 / std)       if std   > 0 else 0.0
    complexity = float(std_d2 / std_d1 / mobility) \
                 if (std_d1 > 0 and mobility > 0) else 0.0

    feat[f"{p}_hjorth_activity"]   = activity
    feat[f"{p}_hjorth_mobility"]   = mobility
    feat[f"{p}_hjorth_complexity"] = complexity

    return feat
