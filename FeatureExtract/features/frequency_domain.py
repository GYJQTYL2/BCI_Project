"""
频域特征提取

每个通道提取以下特征:
    各频段绝对功率  (delta / theta / alpha / beta / gamma)
    各频段相对功率  (band_power / total_power)
    总功率
    功率谱熵
"""

import numpy as np
from scipy.signal import welch

# EEG 标准频段定义 (Hz)
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# beta 子频段，用于 EMG 污染检测
_BETA_LOW  = (13.0, 20.0)
_BETA_HIGH = (20.0, 30.0)
_EPS = 1e-10


def extract(epoch: np.ndarray, ch_name: str, fs: float) -> dict:
    """
    提取单通道单 epoch 的频域特征

    参数:
        epoch   : 一维 numpy 数组
        ch_name : 通道名，用于构造特征列名前缀
        fs      : 采样率（Hz）
    返回:
        {特征名: 特征值} 的字典，坏道（全 NaN）时所有值返回 NaN
    """
    feat = {}
    p = ch_name  # prefix

    # 坏道：全 NaN，返回 NaN 特征让下游 _band_mean 跳过
    if np.all(np.isnan(epoch)):
        for band in BANDS:
            feat[f"{p}_{band}_power"] = np.nan
            feat[f"{p}_{band}_rel"]   = np.nan
        feat[f"{p}_total_power"]      = np.nan
        feat[f"{p}_spectral_entropy"] = np.nan
        return feat

    # Welch 法估计功率谱密度
    # Use full epoch length for 1 Hz frequency resolution (256 samples → 1 Hz bins)
    # Coarser nperseg (e.g. 64) gives 4 Hz bins, leaving delta (0.5-4 Hz) with only one
    # boundary bin, making np.trapz return 0.
    nperseg = len(epoch)
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg)

    # 各频段绝对功率
    band_powers: dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        band_powers[band] = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
        feat[f"{p}_{band}_power"] = band_powers[band]

    # beta EMG 污染检测：用 beta_high/beta_low 比值连续降权
    # 真实 beta 节律符合 1/f 形态，beta_low > beta_high（emg_score < 1）
    # 面部 EMG 宽带平坦，beta_high ≈ beta_low（emg_score > 1.5）
    mask_bl = (freqs >= _BETA_LOW[0])  & (freqs <= _BETA_LOW[1])
    mask_bh = (freqs >= _BETA_HIGH[0]) & (freqs <= _BETA_HIGH[1])
    beta_low_p  = float(np.trapz(psd[mask_bl], freqs[mask_bl])) if mask_bl.any() else 0.0
    beta_high_p = float(np.trapz(psd[mask_bh], freqs[mask_bh])) if mask_bh.any() else 0.0
    emg_score = beta_high_p / (beta_low_p + _EPS)
    # weight: emg_score < 1.5 → 1.0（全保留），emg_score = 3.0 → 0.0（只保留 beta_low）
    weight = float(np.clip(1.0 - (emg_score - 1.5) / 1.5, 0.0, 1.0))
    feat[f"{p}_beta_power"] = beta_low_p + weight * beta_high_p
    band_powers["beta"] = feat[f"{p}_beta_power"]

    # 各频段相对功率
    total = sum(band_powers.values())
    for band, power in band_powers.items():
        feat[f"{p}_{band}_rel"] = power / total if total > 0 else 0.0

    feat[f"{p}_total_power"] = total

    # 功率谱熵
    psd_norm = psd / (psd.sum() + 1e-12)
    feat[f"{p}_spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    return feat
