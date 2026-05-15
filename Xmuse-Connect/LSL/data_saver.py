"""
DataSaver — 分 segment 保存 LSL 采集数据

用法示例：
    from timestamp_corrector import TimestampCorrector

    corrector = TimestampCorrector(inlet, dejitter=True)
    saver = DataSaver(
        save_dir="signal_data/20260427_110000",
        stream_type="EEG",
        ch_names=["TP9", "AF7", "AF8", "TP10"],
        nominal_srate=256.0,
        corrector=corrector,
        segment_seconds=60,       # 按时长分段（优先）
        # segment_bytes=10*1024*1024,  # 或：按文件大小分段（10 MB）
        # segment_samples=15360,       # 或：按样本数分段
    )

    # 采集循环中调用（传入原始 LSL 时间戳，校正在内部完成）：
    saver.add(samples, lsl_timestamps)

    # 结束时：
    saver.close()
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from timestamp_corrector import TimestampCorrector

log = logging.getLogger(__name__)


class DataSaver:
    """
    缓冲 LSL 样本并按 segment 写入 CSV。

    时间戳校正由外部传入的 TimestampCorrector 负责，DataSaver 只管分段与落盘。

    文件命名：{save_dir}/{stream_type}_seg0001.csv, _seg0002.csv, ...

    分段策略（三选一，优先级：segment_seconds > segment_bytes > segment_samples）：
        segment_seconds  — 每段最多包含多少秒的数据（按采样率换算样本数）
        segment_bytes    — 每段 CSV 文件大小上限（字节），按每列 %.6f 格式估算样本数
        segment_samples  — 每段最多包含多少个样本

    注意：segment_bytes 基于每样本字节数的静态估算，实际文件大小可能略有偏差。
    """

    # %.6f 格式下每列（含分隔符）的保守字节估算：
    #   时间戳列约 18 字符，数据列约 15 字符，取 18 作为上界。
    _BYTES_PER_COL_EST: int = 18

    def __init__(
        self,
        save_dir: str,
        stream_type: str,
        ch_names: List[str],
        corrector: TimestampCorrector,
        nominal_srate: float = 256.0,
        segment_seconds: Optional[float] = 60.0,
        segment_bytes: Optional[int] = None,
        segment_samples: Optional[int] = None,
    ):
        self.save_dir    = Path(save_dir)
        self.stream_type = stream_type
        self.ch_names    = ch_names
        self._corrector  = corrector

        # 计算每段样本上限
        if segment_seconds is not None:
            self._seg_limit = int(segment_seconds * nominal_srate)
        elif segment_bytes is not None:
            # (timestamp列 + 数据列) × 每列估算字节数 + 换行符
            n_cols = len(ch_names) + 1
            bytes_per_sample = n_cols * self._BYTES_PER_COL_EST + 1
            self._seg_limit = max(1, segment_bytes // bytes_per_sample)
            log.debug(
                f"[DataSaver/{stream_type}] segment_bytes={segment_bytes}, "
                f"估算每样本 {bytes_per_sample} 字节 → _seg_limit={self._seg_limit} 样本"
            )
        elif segment_samples is not None:
            self._seg_limit = segment_samples
        else:
            self._seg_limit = int(60 * nominal_srate)  # 默认 60 秒

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._buf_samples:    List[List[float]] = []
        self._buf_timestamps: List[float]       = []
        self._seg_index   = 1
        self._total_saved = 0

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def add(self, samples: List[List[float]], lsl_timestamps: List[float]) -> None:
        if not samples:
            return
        self._buf_samples.extend(samples)
        self._buf_timestamps.extend(lsl_timestamps)

        while len(self._buf_samples) >= self._seg_limit:
            self._flush_segment(self._seg_limit)

    def close(self) -> None:
        """将缓冲区剩余数据写入最后一个 segment，完成保存。"""
        if self._buf_samples:
            self._flush_segment(len(self._buf_samples))
        log.info(
            f"[DataSaver/{self.stream_type}] 保存完毕，"
            f"共 {self._seg_index - 1} 个 segment，{self._total_saved} 个样本"
        )

    @property
    def buffered(self) -> int:
        """当前缓冲区中未写入的样本数。"""
        return len(self._buf_samples)

    @property
    def total_saved(self) -> int:
        """已写入磁盘的累计样本数（不含缓冲区）。"""
        return self._total_saved

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _flush_segment(self, n: int) -> None:
        """取出缓冲区前 n 个样本，写入当前 segment 文件。"""
        samples    = self._buf_samples[:n]
        timestamps = self._buf_timestamps[:n]
        self._buf_samples    = self._buf_samples[n:]
        self._buf_timestamps = self._buf_timestamps[n:]

        path = self._seg_path(self._seg_index)
        self._write_csv(path, samples, timestamps)

        self._total_saved += n
        log.info(
            f"[DataSaver/{self.stream_type}] segment {self._seg_index:04d} "
            f"→ {path.name}  ({n} 样本，累计 {self._total_saved})"
        )
        self._seg_index += 1

    def _seg_path(self, index: int) -> Path:
        return self.save_dir / f"{self.stream_type}_seg{index:04d}.csv"

    def _write_csv(
        self,
        path: Path,
        samples: List[List[float]],
        timestamps: List[float],
    ) -> None:
        try:
            data = np.c_[np.array(timestamps), np.array(samples)]
            pd.DataFrame(data, columns=["timestamp"] + self.ch_names).to_csv(
                path, float_format="%.6f", index=False
            )
        except Exception:
            log.exception(f"[DataSaver/{self.stream_type}] 写入 {path} 失败")
