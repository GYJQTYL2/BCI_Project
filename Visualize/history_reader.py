"""
历史数据读取器

从 CSV segment 文件中读取历史 EEG/特征/注意力数据，
支持按时间范围筛选，输出格式与 DataBridge.snapshot() 对应字段保持一致，
前端可直接复用实时图表组件渲染历史数据。
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# session 目录命名格式：YYYYMMDD_HHMMSS
_SESSION_RE = re.compile(r"^\d{8}_\d{6}$")

# attention level 整数 → 字符串
_INT_TO_LEVEL = {0: "low", 1: "medium", 2: "high"}

# 各数据类型的 segment 文件名前缀
_FILE_PREFIX: Dict[str, str] = {
    "raw":       "EEG",
    "processed": "EEG_processed",
    "features":  "features",
    "attention": "attention",
}

# 特征面板只展示频段功率，通道与频段名称
_FEATURE_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
_FEATURE_BANDS    = ["delta", "theta", "alpha", "beta", "gamma"]


# 同一采集会话各数据类型的目录创建时间可能相差数秒，视为同一 session
_SESSION_MERGE_THRESHOLD = 120   # 秒


# ── 公开接口 ──────────────────────────────────────────────────────────────────

class HistoryReader:
    """
    历史 CSV 数据读取器

    参数:
        data_dirs : 各数据类型的根目录映射，键为数据类型名称
                    {
                        "raw":       "/path/to/signal_data",
                        "processed": "/path/to/processed_data",
                        "features":  "/path/to/features_output",
                        "attention": "/path/to/attention_output",
                    }
                    可只提供部分键；缺省的类型在 API 层返回空结果。
    """

    def __init__(self, data_dirs: Dict[str, str]):
        self._dirs: Dict[str, Path] = {k: Path(v) for k, v in data_dirs.items()}

    # ── 列出可用 session ──────────────────────────────────────────────────

    def list_sessions(self) -> List[dict]:
        """
        扫描所有数据目录，返回可用 session 列表（按时间降序）。

        返回示例:
            [
              {
                "session_id": "20260511_120000",
                "display":    "2026-05-11 12:00:00",
                "types":      ["raw", "processed", "features", "attention"],
                "start_ts":   1746950400.0,
                "end_ts":     1746954000.0,
              },
              ...
            ]
        """
        sessions: Dict[str, dict] = {}

        for dtype, base in self._dirs.items():
            if not base.exists():
                continue
            for sub in sorted(base.iterdir()):
                if not sub.is_dir() or not _SESSION_RE.match(sub.name):
                    continue
                sid = sub.name
                # 查找是否能合并到已有 session（相差 < 阈值秒）
                canonical = _find_canonical_session(sessions, sid)
                if canonical is None:
                    canonical = sid
                    sessions[sid] = {
                        "session_id": sid,
                        "display":    _session_display(sid),
                        "types":      [],
                        "start_ts":   None,
                        "end_ts":     None,
                    }
                t0, t1 = _session_time_range(sub, _FILE_PREFIX[dtype])
                # 只在目录内确实有 segment 文件时才将该类型纳入 types
                if t0 is not None:
                    sessions[canonical]["types"].append(dtype)
                    if sessions[canonical]["start_ts"] is None or t0 < sessions[canonical]["start_ts"]:
                        sessions[canonical]["start_ts"] = t0
                if t1 is not None:
                    if sessions[canonical]["end_ts"] is None or t1 > sessions[canonical]["end_ts"]:
                        sessions[canonical]["end_ts"] = t1

        return sorted(sessions.values(), key=lambda s: s["session_id"], reverse=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────

    def load(
        self,
        data_type:  str,
        session_id: str,
        start_ts:   Optional[float] = None,
        end_ts:     Optional[float] = None,
        max_points: int = 3000,
    ) -> dict:
        """
        读取指定 session 的数据，按时间范围筛选并下采样后返回。

        返回格式与 DataBridge.snapshot() 中对应字段一致：
            raw / processed → {"timestamps": [...], "ch_1": [...], ...}
            features        → {"timestamps": [...], "CH1": {"delta": [...], ...}, ...}
            attention       → {"timestamps": [...], "attention_score": [...], "level": [...], ...}
        """
        if data_type not in self._dirs:
            log.warning(f"[HistoryReader] 未配置数据类型: {data_type}")
            return {}

        session_dir = _find_session_dir(self._dirs[data_type], session_id)
        if session_dir is None:
            log.warning(f"[HistoryReader] 未找到 session: {data_type}/{session_id} (±{_SESSION_MERGE_THRESHOLD}s)")
            return {}

        df = _load_segments(session_dir, _FILE_PREFIX[data_type])
        if df is None or df.empty:
            return {}

        # LSL 本地时钟 → Unix 时间戳
        # 用 canonical session_id 而非实际目录名作锚点，确保同一会话各类型时间轴对齐。
        # 各类型目录创建时间相差 1~5 秒，若用各自目录名会产生等量的系统性偏移。
        unix_offset = _lsl_unix_offset(session_id, float(df["timestamp"].iloc[0]))
        df["timestamp"] = df["timestamp"] + unix_offset

        # 时间范围过滤（现在 timestamp 列已是 Unix 时间戳）
        if start_ts is not None:
            df = df[df["timestamp"] >= start_ts]
        if end_ts is not None:
            df = df[df["timestamp"] <= end_ts]
        if df.empty:
            return {}

        # 均匀下采样（保留首尾）
        if len(df) > max_points:
            idx = np.round(np.linspace(0, len(df) - 1, max_points)).astype(int)
            df = df.iloc[idx].copy()

        ts_fmt = _fmt_ts(df["timestamp"].tolist())

        if data_type in ("raw", "processed"):
            return _to_eeg_dict(df, ts_fmt)
        elif data_type == "features":
            return _to_features_dict(df, ts_fmt)
        elif data_type == "attention":
            return _to_attention_dict(df, ts_fmt)
        return {}


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _sid_to_ts(sid: str) -> float:
    """session_id 字符串 → Unix 时间戳"""
    try:
        return datetime.datetime.strptime(sid, "%Y%m%d_%H%M%S").timestamp()
    except Exception:
        return 0.0


def _find_canonical_session(sessions: dict, sid: str) -> Optional[str]:
    """
    在已有 sessions 中查找与 sid 时间差小于阈值的 canonical session_id。
    原始 EEG 与处理后数据的目录创建时间可能相差 1~10 秒。
    """
    target = _sid_to_ts(sid)
    for key in sessions:
        if abs(_sid_to_ts(key) - target) < _SESSION_MERGE_THRESHOLD:
            return key
    return None


def _find_session_dir(base: Path, session_id: str) -> Optional[Path]:
    """
    在 base 下查找 session_id 对应的目录，允许时间戳偏差 < 阈值秒。
    """
    exact = base / session_id
    if exact.exists():
        return exact
    if not base.exists():
        return None
    target = _sid_to_ts(session_id)
    if target == 0.0:
        return None
    best: Optional[Path] = None
    best_diff = _SESSION_MERGE_THRESHOLD
    for sub in base.iterdir():
        if not sub.is_dir() or not _SESSION_RE.match(sub.name):
            continue
        diff = abs(_sid_to_ts(sub.name) - target)
        if diff < best_diff:
            best_diff, best = diff, sub
    return best


def _session_display(sid: str) -> str:
    """'20260511_120000' → '2026-05-11 12:00:00'"""
    try:
        return f"{sid[:4]}-{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}:{sid[13:15]}"
    except Exception:
        return sid


def _session_time_range(
    session_dir: Path,
    prefix: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    快速获取 session 的时间范围，返回 Unix 时间戳。

    CSV 中保存的是 LSL 本地时钟（monotonic），不是 Unix 时间戳。
    用 session 文件夹名（由 datetime.now() 生成，挂钟时间）作为锚点估算偏移量。
    """
    segs = sorted(session_dir.glob(f"{prefix}_seg*.csv"))
    if not segs:
        return None, None
    try:
        first_lsl = float(
            pd.read_csv(segs[0], usecols=["timestamp"], nrows=1)["timestamp"].iloc[0]
        )
        last_lsl = float(
            pd.read_csv(segs[-1], usecols=["timestamp"])["timestamp"].iloc[-1]
        )
        offset = _lsl_unix_offset(session_dir.name, first_lsl)
        return first_lsl + offset, last_lsl + offset
    except Exception:
        log.debug(f"[HistoryReader] 无法读取时间范围: {session_dir}")
        return None, None


def _load_segments(session_dir: Path, prefix: str) -> Optional[pd.DataFrame]:
    """加载目录下所有 segment CSV 并按时间戳合并"""
    segs = sorted(session_dir.glob(f"{prefix}_seg*.csv"))
    if not segs:
        log.warning(f"[HistoryReader] 未找到 {prefix}_seg*.csv in {session_dir}")
        return None
    try:
        df = pd.concat([pd.read_csv(f) for f in segs], ignore_index=True)
        df.sort_values("timestamp", inplace=True, ignore_index=True)
        return df
    except Exception:
        log.exception(f"[HistoryReader] 读取失败: {session_dir}")
        return None


def _lsl_unix_offset(session_id: str, first_lsl_ts: float) -> float:
    """
    估算 LSL 本地时钟 → Unix 时间戳的偏移量。

    session 文件夹名由 datetime.now().strftime("%Y%m%d_%H%M%S") 生成（挂钟本地时间），
    将其转为 Unix 时间戳后减去 CSV 中第一个 LSL 时间戳，即得偏移量。
    实际数据帧比文件夹创建时刻略晚（通常 < 5 秒），误差可接受。
    """
    try:
        dt = datetime.datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        return dt.timestamp() - first_lsl_ts
    except Exception:
        return 0.0


def _fmt_ts(timestamps: list) -> list:
    """Unix 时间戳列表 → 'HH:MM:SS.mmm' 字符串列表"""
    result = []
    for ts in timestamps:
        dt = datetime.datetime.fromtimestamp(float(ts))
        result.append(dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}")
    return result


def _to_eeg_dict(df: pd.DataFrame, ts_fmt: list) -> dict:
    """raw / processed → {"timestamps": [...], "ch_1": [...], ...}"""
    ch_cols = [c for c in df.columns if c != "timestamp"]
    return {"timestamps": ts_fmt, **{c: df[c].tolist() for c in ch_cols}}


def _to_features_dict(df: pd.DataFrame, ts_fmt: list) -> dict:
    """features → {"timestamps": [...], "CH1": {"delta": [...], ...}, ...}"""
    result: dict = {"timestamps": ts_fmt}
    for ch in _FEATURE_CHANNELS:
        ch_data = {}
        for band in _FEATURE_BANDS:
            col = f"{ch}_{band}_power"
            if col in df.columns:
                ch_data[band] = df[col].tolist()
        if ch_data:
            result[ch] = ch_data
    return result


def _to_attention_dict(df: pd.DataFrame, ts_fmt: list) -> dict:
    """attention → {"timestamps": [...], "attention_score": [...], "level": [...], ...}"""
    result: dict = {"timestamps": ts_fmt}
    for col in ("attention_score", "engagement_index", "theta_alpha_ratio"):
        if col in df.columns:
            result[col] = df[col].tolist()
    if "level" in df.columns:
        result["level"] = [_INT_TO_LEVEL.get(int(round(float(v))), "low") for v in df["level"]]
    return result
