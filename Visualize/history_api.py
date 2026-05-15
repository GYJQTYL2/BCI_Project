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
from typing import Optional

from aiohttp import web

from history_reader import HistoryReader

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
        log.info("[HistoryAPI] 路由已注册: /history/sessions  /history/data")

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


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

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
