"""
历史数据 HTTP API

向 aiohttp 应用注册两个只读端点：
  GET /history/sessions
      → 返回所有可用 session 列表（JSON）

  GET /history/data?type=&session=&start=&end=&max_points=
      type       : raw | processed | features | attention
      session    : YYYYMMDD_HHMMSS
      start      : 起始 Unix 时间戳（浮点，可选）
      end        : 结束 Unix 时间戳（浮点，可选）
      max_points : 最大返回数据点数，默认 3000
      → 返回与 DataBridge.snapshot() 对应字段格式相同的 JSON
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from aiohttp import web

from history_reader import HistoryReader

_REPORT_DIR = Path(__file__).parent.parent / "Report"
sys.path.insert(0, str(_REPORT_DIR))
from report_generator import EduReportGenerator

_EM_DIR = Path(__file__).parent.parent / "EventMarker"
sys.path.insert(0, str(_EM_DIR))
try:
    from marker_store import MarkerStore as _EventMarkerStore
    _EM_AVAILABLE = True
except ImportError:
    _EM_AVAILABLE = False

log = logging.getLogger(__name__)


class HistoryAPI:
    """
    历史数据 API 模块

    参数:
        reader : HistoryReader 实例
    """

    def __init__(self, reader: HistoryReader):
        self._reader = reader

    def register_routes(self, app: web.Application) -> None:
        """向 aiohttp Application 注册 /history/* 路由"""
        app.router.add_get("/history/sessions", self._handle_sessions)
        app.router.add_get("/history/data",     self._handle_data)
        app.router.add_get("/history/report",   self._handle_report)
        log.info("[HistoryAPI] 路由已注册: /history/sessions  /history/data  /history/report")

    # ── 请求处理 ──────────────────────────────────────────────────────────

    async def _handle_sessions(self, request: web.Request) -> web.Response:
        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(None, self._reader.list_sessions)
        return _json_response(sessions)

    async def _handle_data(self, request: web.Request) -> web.Response:
        q          = request.rel_url.query
        dtype      = q.get("type", "").strip()
        session_id = q.get("session", "").strip()
        start_ts   = _to_float(q.get("start"))
        end_ts     = _to_float(q.get("end"))
        max_points = int(q.get("max_points", 3000))

        if not dtype or not session_id:
            return web.Response(status=400, text="type 和 session 参数必填")

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: self._reader.load(dtype, session_id, start_ts, end_ts, max_points),
        )
        return _json_response(data)


    async def _handle_report(self, request: web.Request) -> web.Response:
        q          = request.rel_url.query
        session_id = q.get("session", "").strip()
        start_ts   = _to_float(q.get("start"))
        end_ts     = _to_float(q.get("end"))

        if not session_id:
            return web.Response(status=400, text="session 参数必填")

        loop = asyncio.get_event_loop()

        def _build():
            attn = self._reader.load("attention",      session_id, start_ts, end_ts, max_points=5000)
            cl   = self._reader.load("cognitive_load", session_id, start_ts, end_ts, max_points=5000)
            report = EduReportGenerator().generate(attn, cl, session_id)
            # 附加热力图所需的原始时间序列
            if attn.get("timestamps"):
                report["attention_series"] = {
                    "timestamps":     attn["timestamps"],
                    "attention_score": attn.get("attention_score", []),
                }
            if cl.get("timestamps"):
                report["cl_series"] = {
                    "timestamps":     cl["timestamps"],
                    "cog_load_score": cl.get("cog_load_score", []),
                }
            # 附加 EventMarker 内容打点
            if _EM_AVAILABLE:
                try:
                    em = _find_em_session(_EventMarkerStore(), session_id)
                    if em:
                        report["markers"]        = em.get("markers", [])
                        report["marker_session"] = {
                            "start_ts": em.get("start_ts"),
                            "end_ts":   em.get("end_ts"),
                        }
                except Exception:
                    log.debug("EventMarker 加载失败，跳过打点数据")
            return report

        report = await loop.run_in_executor(None, _build)
        return _json_response(report)


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

def _find_em_session(store, session_id: str):
    """按 session_id 精确匹配或按时间接近度（±5 分钟）模糊匹配 EventMarker 会话。"""
    from datetime import datetime
    rec = store.get_session(session_id)
    if rec:
        return rec
    try:
        target = datetime.strptime(session_id, "%Y%m%d_%H%M%S").timestamp()
        for s in store.list_sessions():
            sid = s.get("session_id", "")
            try:
                ts = datetime.strptime(sid, "%Y%m%d_%H%M%S").timestamp()
                if abs(ts - target) < 300:
                    return store.get_session(sid)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _to_float(v: Optional[str]) -> Optional[float]:
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _json_response(data) -> web.Response:
    return web.Response(
        text=json.dumps(data, ensure_ascii=False, allow_nan=False),
        content_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )
