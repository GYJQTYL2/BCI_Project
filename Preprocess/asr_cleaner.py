"""
ASR (Artifact Subspace Reconstruction) 封装

用法：
    1. 录制基线（安静闭眼 30 秒）
       cleaner = ASRCleaner()
       cleaner.record_baseline(df)   # 喂入基线 DataFrame
       cleaner.save_baseline("baseline.pkl")

    2. 下次直接加载
       cleaner = ASRCleaner()
       cleaner.load_baseline("baseline.pkl")

    3. 实时清理
       df_clean = cleaner.clean(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
_DEFAULT_BASELINE = Path(__file__).parent / "asr_baseline.pkl"


class ASRCleaner:
    """
    封装 asrpy.ASR，管理基线训练、保存/加载和实时清理。

    参数:
        sfreq   : 采样率，默认 256 Hz
        cutoff  : ASR 阈值（标准差倍数），越小越激进。
                  20 = 只去大伪迹（推荐起始值），10 = 更严格
        baseline_path : 默认基线文件路径
    """

    def __init__(
        self,
        sfreq: float = 256.0,
        cutoff: float = 20.0,
        baseline_path: str | Path = _DEFAULT_BASELINE,
    ):
        from asrpy import ASR
        self._sfreq = sfreq
        self._cutoff = cutoff
        self._baseline_path = Path(baseline_path)
        self._asr: Optional[ASR] = None
        self._baseline_buf: list[np.ndarray] = []  # 录制基线时的累积缓冲

    # ── 基线管理 ──────────────────────────────────────────────────────────

    def start_recording(self) -> None:
        """开始录制基线，清空缓冲区"""
        self._baseline_buf = []

    def feed_baseline(self, df: pd.DataFrame) -> None:
        """喂入一帧基线数据（DataFrame，含 CH1–CH4 列）"""
        data = df[_CHANNELS].values.T  # (4, n_samples)
        self._baseline_buf.append(data)

    def finish_recording(self) -> int:
        """
        用录制到的数据训练 ASR。
        返回基线样本总数，< 7680（30 秒）时会发出警告。
        """
        if not self._baseline_buf:
            raise RuntimeError("未录制到任何基线数据，请先调用 start_recording/feed_baseline")

        from asrpy import ASR
        baseline = np.concatenate(self._baseline_buf, axis=1)  # (4, total_samples)
        n_samples = baseline.shape[1]

        if n_samples < int(self._sfreq * 30):
            print(f"[ASR] 警告：基线长度 {n_samples/self._sfreq:.1f}s，建议 >= 30s")

        import mne
        info = mne.create_info(ch_names=_CHANNELS, sfreq=self._sfreq, ch_types="eeg")
        raw = mne.io.RawArray(baseline, info, verbose=False)

        self._asr = ASR(sfreq=self._sfreq, cutoff=self._cutoff)
        self._asr.fit(raw)
        self._baseline_buf = []
        print(f"[ASR] 训练完成，基线 {n_samples/self._sfreq:.1f}s ({n_samples} 样本)")
        return n_samples

    def save_baseline(self, path: str | Path | None = None) -> Path:
        """保存训练好的 ASR 状态到文件"""
        if self._asr is None:
            raise RuntimeError("ASR 尚未训练")
        import pickle
        save_path = Path(path) if path else self._baseline_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self._asr, f)
        print(f"[ASR] 基线已保存 → {save_path}")
        return save_path

    def load_baseline(self, path: str | Path | None = None) -> bool:
        """
        从文件加载 ASR 状态。
        返回 True = 加载成功，False = 文件不存在。
        """
        import pickle
        load_path = Path(path) if path else self._baseline_path
        if not load_path.exists():
            return False
        with open(load_path, "rb") as f:
            self._asr = pickle.load(f)
        print(f"[ASR] 基线已加载 ← {load_path}")
        return True

    @property
    def is_ready(self) -> bool:
        return self._asr is not None

    @property
    def baseline_path(self) -> Path:
        return self._baseline_path

    # ── 实时清理 ──────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对一帧 EEG 数据应用 ASR 去伪迹。
        若 ASR 未训练，原样返回。
        """
        if self._asr is None:
            return df

        data = df[_CHANNELS].values.T  # (4, n_samples)
        try:
            import mne
            info = mne.create_info(ch_names=_CHANNELS, sfreq=self._sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data, info, verbose=False)
            clean_raw = self._asr.transform(raw)
            clean = clean_raw.get_data()  # (4, n_samples)
        except Exception as e:
            print(f"[ASR] transform 失败，跳过本窗口: {e}")
            return df

        df_out = df.copy()
        for i, ch in enumerate(_CHANNELS):
            df_out[ch] = clean[i]
        return df_out
