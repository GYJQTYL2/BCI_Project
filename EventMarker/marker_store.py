"""
内容打点数据解析与存储

接收第三方发送的会话信息（start_time, end_time, markers），
解析并保存为 JSON + CSV 两种格式。

time_range 支持以下格式：
  - [start, end]                         绝对时间戳或相对秒数
  - {"start": ..., "end": ...}           字典，键名还支持 from/to/begin/stop
  - 值类型：Unix 时间戳（float/int）、ISO 字符串、HH:MM:SS 字符串
  - 若值 < 1e8 且提供了 session_start_ts，自动视为相对秒数转换为绝对时间戳
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent / "marker_data"

_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%H:%M:%S.%f",
    "%H:%M:%S",
)

_MARKER_CSV_FIELDS = ["id", "content", "start_str", "end_str", "duration_s", "start_ts", "end_ts"]


# ── 时间解析工具 ──────────────────────────────────────────────────────────────

def _parse_ts(v: Any) -> Optional[float]:
    """将各种格式的时间值转为 Unix 时间戳（float）"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        for fmt in _TS_FORMATS:
            try:
                return datetime.strptime(v, fmt).timestamp()
            except ValueError:
                continue
        raise ValueError(f"无法解析时间字符串: {v!r}")
    raise ValueError(f"不支持的时间类型: {type(v).__name__}")


def _ts_to_str(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _parse_time_range(raw: Any, session_start_ts: Optional[float] = None) -> Dict:
    """
    解析 time_range 字段，返回标准化字典。

    支持输入:
        [start, end]
        {"start": ..., "end": ...} / {"from": ..., "to": ...}
    """
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        raw_start, raw_end = raw[0], raw[1]
    elif isinstance(raw, dict):
        raw_start = raw.get("start") or raw.get("from") or raw.get("begin")
        raw_end   = raw.get("end")   or raw.get("to")   or raw.get("stop")
    else:
        raise ValueError(f"无法解析 time_range: {raw!r}")

    start_ts = _parse_ts(raw_start)
    end_ts   = _parse_ts(raw_end)

    # 值较小时视为相对于会话开始的秒数偏移
    if session_start_ts is not None:
        if start_ts is not None and start_ts < 1e8:
            start_ts += session_start_ts
        if end_ts is not None and end_ts < 1e8:
            end_ts += session_start_ts

    duration = round(end_ts - start_ts, 3) if (start_ts is not None and end_ts is not None) else None

    return {
        "start_ts":   start_ts,
        "end_ts":     end_ts,
        "start_str":  _ts_to_str(start_ts),
        "end_str":    _ts_to_str(end_ts),
        "duration_s": duration,
    }


# ── 核心存储类 ────────────────────────────────────────────────────────────────

class MarkerStore:
    """
    内容打点数据存储管理器

    参数:
        output_dir : 数据保存根目录，默认 EventMarker/marker_data/
    """

    def __init__(self, output_dir: Union[str, Path, None] = None):
        self._root = Path(output_dir) if output_dir else _DEFAULT_OUTPUT
        self._root.mkdir(parents=True, exist_ok=True)
        log.info(f"[MarkerStore] 数据目录: {self._root.resolve()}")

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def save_session(self, data: dict) -> dict:
        """
        解析并保存完整会话。

        必填字段:
            start_time  会话开始时间
        常用字段:
            end_time    会话结束时间
            session_id  自定义 ID；缺省时由 start_time 自动生成
            markers     内容打点列表
        """
        start_ts = _parse_ts(data.get("start_time"))
        if start_ts is None:
            raise ValueError("缺少必填字段: start_time")

        end_ts = _parse_ts(data.get("end_time"))

        session_id  = (data.get("session_id") or "").strip()
        if not session_id:
            session_id = datetime.fromtimestamp(start_ts).strftime("%Y%m%d_%H%M%S")

        session_dir = self._root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        markers = self._parse_markers(data.get("markers") or [], start_ts)

        record = {
            "session_id":       session_id,
            "received_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_time":       _ts_to_str(start_ts),
            "start_ts":         start_ts,
            "end_time":         _ts_to_str(end_ts),
            "end_ts":           end_ts,
            "duration_seconds": round(end_ts - start_ts, 3) if end_ts else None,
            "marker_count":     len(markers),
            "markers":          markers,
        }

        self._write_json(session_dir / "session.json", record)
        if markers:
            self._write_csv(session_dir / "markers.csv", markers, append=False)

        log.info(f"[MarkerStore] 保存会话 {session_id}，{len(markers)} 个打点 → {session_dir}")
        return {
            "session_id":   session_id,
            "marker_count": len(markers),
            "saved_to":     str(session_dir),
        }

    def append_marker(self, session_id: str, marker_data: dict) -> dict:
        """向已有会话追加单个打点（支持增量接收）。"""
        session_dir = self._root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        json_path = session_dir / "session.json"
        if json_path.exists():
            record = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            record = self._empty_record(session_id)

        parsed = self._parse_markers([marker_data], record.get("start_ts"))
        record["markers"].extend(parsed)
        record["marker_count"] = len(record["markers"])

        # 重新分配连续 id
        for i, m in enumerate(record["markers"]):
            m["id"] = i

        self._write_json(json_path, record)

        csv_path  = session_dir / "markers.csv"
        first_row = not csv_path.exists()
        self._write_csv(csv_path, parsed, append=not first_row)

        log.info(f"[MarkerStore] 追加打点到 {session_id}，共 {record['marker_count']} 个")
        return {"session_id": session_id, "marker_count": record["marker_count"]}

    def update_session_time(self, session_id: str, start_time=None, end_time=None) -> dict:
        """更新已有会话的 start_time / end_time。"""
        json_path = self._root / session_id / "session.json"
        if not json_path.exists():
            raise FileNotFoundError(f"会话不存在: {session_id}")

        record = json.loads(json_path.read_text(encoding="utf-8"))

        if start_time is not None:
            ts = _parse_ts(start_time)
            record["start_ts"]   = ts
            record["start_time"] = _ts_to_str(ts)
        if end_time is not None:
            ts = _parse_ts(end_time)
            record["end_ts"]   = ts
            record["end_time"] = _ts_to_str(ts)

        s, e = record.get("start_ts"), record.get("end_ts")
        record["duration_seconds"] = round(e - s, 3) if (s and e) else None

        self._write_json(json_path, record)
        return {"session_id": session_id, "updated": True}

    def get_session(self, session_id: str) -> Optional[dict]:
        """读取指定会话的完整记录。"""
        path = self._root / session_id / "session.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> List[dict]:
        """列出所有已保存会话（按时间降序）。"""
        sessions = []
        for d in sorted(self._root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            jp = d / "session.json"
            if not jp.exists():
                continue
            try:
                rec = json.loads(jp.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id":   rec.get("session_id", d.name),
                    "start_time":   rec.get("start_time"),
                    "end_time":     rec.get("end_time"),
                    "duration_seconds": rec.get("duration_seconds"),
                    "marker_count": rec.get("marker_count", 0),
                    "received_at":  rec.get("received_at"),
                })
            except Exception:
                log.debug(f"[MarkerStore] 跳过无效会话目录: {d.name}")
        return sessions

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _parse_markers(self, raw_list: list, session_start_ts: Optional[float]) -> List[dict]:
        markers = []
        for idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                log.warning(f"[MarkerStore] 跳过无效打点 #{idx}: {item!r}")
                continue
            try:
                tr_raw = item.get("time_range") or item.get("timeRange") or item.get("range")
                tr = _parse_time_range(tr_raw, session_start_ts) if tr_raw is not None else {}
            except Exception as e:
                log.warning(f"[MarkerStore] 打点 #{idx} time_range 解析失败: {e}")
                tr = {}

            marker: dict = {
                "id":      idx,
                "content": item.get("content") or item.get("label") or item.get("text") or "",
                **tr,
            }
            # 保留原始附加字段
            skip = {"time_range", "timeRange", "range", "content", "label", "text"}
            for k, v in item.items():
                if k not in skip:
                    marker[k] = v

            markers.append(marker)
        return markers

    @staticmethod
    def _empty_record(session_id: str) -> dict:
        return {
            "session_id":       session_id,
            "received_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_time":       None,
            "start_ts":         None,
            "end_time":         None,
            "end_ts":           None,
            "duration_seconds": None,
            "marker_count":     0,
            "markers":          [],
        }

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_csv(path: Path, markers: List[dict], append: bool) -> None:
        # 动态收集所有字段名（保持顺序，固定字段优先）
        extra_keys: List[str] = []
        for m in markers:
            for k in m:
                if k not in _MARKER_CSV_FIELDS and k not in extra_keys:
                    extra_keys.append(k)
        fieldnames = _MARKER_CSV_FIELDS + extra_keys

        mode = "a" if append else "w"
        with open(path, mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not append:
                writer.writeheader()
            writer.writerows(markers)
