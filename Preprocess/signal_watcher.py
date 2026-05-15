"""
Xmuse-Connect LSL signal_data 目录监控与自动预处理

监控目录结构:
    signal_data/
    └── YYYYMMDD_HHMMSS/        ← session 目录（采集开始时创建）
        ├── EEG_seg0001.csv     ← segment 文件（持续生成）
        ├── EEG_seg0002.csv
        └── ...

segment 完成判定（两种方式任一满足）:
    1. 更新的 segment 已存在（EEG_seg0002 存在 → EEG_seg0001 已完整）
    2. 跨两次轮询文件大小未变化（用于 session 最后一个 segment）

断点续处理:
    已处理记录持久化到 output_dir/.processed_files，重启后自动跳过。

列名适配 (signal_data CSV → pipeline 期望格式):
    timestamp → timestamps
    ch_1~ch_4 → eeg_1~eeg_4
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eeg_pipeline import EEGPreprocessPipeline

log = logging.getLogger(__name__)

# signal_data CSV 列名 → data_clean 期望列名
_COL_ADAPT: Dict[str, str] = {
    "timestamp": "timestamps",
    "ch_1": "eeg_1",
    "ch_2": "eeg_2",
    "ch_3": "eeg_3",
    "ch_4": "eeg_4",
}


class SignalDataWatcher:
    """
    持续监控 signal_data 目录，对完成写入的 EEG segment 自动预处理。

    用法:
        watcher = SignalDataWatcher(
            signal_data_dir="Xmuse-Connect/LSL/signal_data",
            output_dir="preprocessed",
        )
        watcher.watch()          # 持续监控，Ctrl+C 停止
        watcher.scan()           # 单次扫描处理
    """

    SEG_GLOB = "EEG_seg*.csv"
    STATE_FILE = ".processed_files"

    def __init__(
        self,
        signal_data_dir: str | Path,
        output_dir: str | Path,
        pipeline: Optional[EEGPreprocessPipeline] = None,
        poll_interval: float = 600.0,
    ):
        """
        参数:
            signal_data_dir : signal_data 根目录路径
            output_dir      : 预处理结果输出目录
            pipeline        : 自定义 EEGPreprocessPipeline 实例，默认使用默认参数
            poll_interval   : 轮询间隔（秒），默认 3s
        """
        self.signal_data_dir = Path(signal_data_dir)
        self.output_dir = Path(output_dir)
        self.pipeline = pipeline or EEGPreprocessPipeline()
        self.poll_interval = poll_interval

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.output_dir / self.STATE_FILE
        self._processed: Set[str] = self._load_state()

        # 跨轮询的文件大小记录，用于判定最后一个 segment 是否稳定
        self._prev_sizes: Dict[str, int] = {}

    # ── 状态持久化 ─────────────────────────────────────────────────────────

    def _load_state(self) -> Set[str]:
        if self._state_path.exists():
            lines = self._state_path.read_text(encoding="utf-8").splitlines()
            return set(line for line in lines if line.strip())
        return set()

    def _mark_processed(self, seg_path: Path) -> None:
        key = self._key(seg_path)
        self._processed.add(key)
        with self._state_path.open("a", encoding="utf-8") as f:
            f.write(key + "\n")

    def _key(self, seg_path: Path) -> str:
        """用相对路径作为去重 key"""
        return str(seg_path.relative_to(self.signal_data_dir))

    def _is_processed(self, seg_path: Path) -> bool:
        return self._key(seg_path) in self._processed

    # ── segment 完成判定 ────────────────────────────────────────────────────

    def _seg_index(self, path: Path) -> int:
        """EEG_seg0003.csv → 3"""
        return int(path.stem.rsplit("seg", 1)[1])

    def _next_seg_path(self, path: Path) -> Path:
        prefix = path.stem.rsplit("seg", 1)[0]  # "EEG_"
        return path.parent / f"{prefix}seg{self._seg_index(path) + 1:04d}.csv"

    def _is_complete(self, seg_path: Path) -> bool:
        """
        判断 segment 是否已完整写入:
          1. 下一个 segment 已存在（最可靠）
          2. 文件大小与上次轮询相同且非空（用于最后一个 segment）
        """
        if self._next_seg_path(seg_path).exists():
            return True

        key = self._key(seg_path)
        try:
            current_size = seg_path.stat().st_size
        except FileNotFoundError:
            return False

        prev_size = self._prev_sizes.get(key)
        self._prev_sizes[key] = current_size
        return prev_size is not None and prev_size == current_size and current_size > 0

    # ── 列名适配 ────────────────────────────────────────────────────────────

    @staticmethod
    def _adapt_columns(df: pd.DataFrame) -> pd.DataFrame:
        """将 signal_data 的列名映射为 pipeline 期望的格式"""
        return df.rename(columns={k: v for k, v in _COL_ADAPT.items() if k in df.columns})

    # ── 单文件处理 ──────────────────────────────────────────────────────────

    def _process(self, seg_path: Path) -> None:
        session_name = seg_path.parent.name          # YYYYMMDD_HHMMSS
        session_out = self.output_dir / session_name
        stem = seg_path.stem                          # EEG_seg0001

        log.info(f"处理: {session_name}/{seg_path.name}")

        df_raw = pd.read_csv(seg_path, na_values=[""])
        df = self._adapt_columns(df_raw)

        self.pipeline.run_df(df, stem, str(session_out))
        self._mark_processed(seg_path)

    # ── 扫描与主循环 ────────────────────────────────────────────────────────

    def scan(self) -> int:
        """
        扫描一次 signal_data 目录，处理所有已完成但未处理的 segment。
        返回本次处理的文件数量。
        """
        count = 0
        for session_dir in sorted(self.signal_data_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for seg_file in sorted(session_dir.glob(self.SEG_GLOB)):
                if self._is_processed(seg_file):
                    continue
                if not self._is_complete(seg_file):
                    continue
                try:
                    self._process(seg_file)
                    count += 1
                except Exception:
                    log.exception(f"处理失败: {seg_file}")
        return count

    def watch(self) -> None:
        """持续轮询监控，Ctrl+C 停止"""
        print(f"监控目录 : {self.signal_data_dir}")
        print(f"输出目录 : {self.output_dir}")
        print(f"轮询间隔 : {self.poll_interval}s  (Ctrl+C 停止)\n")
        try:
            while True:
                n = self.scan()
                if n:
                    print(f"[{_ts()}] 本轮处理 {n} 个 segment")
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n监控已停止")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── CLI 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG signal_data 自动预处理监控")
    parser.add_argument("signal_dir", help="signal_data 目录路径")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录（默认: signal_data/../preprocessed）",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float, default=600.0,
        help="轮询间隔秒数（默认 600，即 10 分钟）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只扫描一次后退出（不持续监控）",
    )
    args = parser.parse_args()

    signal_dir = Path(args.signal_dir)
    output_dir = Path(args.output) if args.output else signal_dir.parent / "preprocessed"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    watcher = SignalDataWatcher(
        signal_data_dir=signal_dir,
        output_dir=output_dir,
        poll_interval=args.interval,
    )

    if args.once:
        n = watcher.scan()
        print(f"处理完成：{n} 个 segment")
    else:
        watcher.watch()
