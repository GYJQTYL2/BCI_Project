"""
内容打点 HTTP 接收服务

监听并接收第三方发来的会话信息和内容打点，解析后保存到文件。

── 接口 ───────────────────────────────────────────────────────────────────────

POST /api/session
    接收完整会话（start_time + end_time + markers[]）
    请求体示例:
    {
        "session_id": "optional_custom_id",
        "start_time": "2026-05-26T14:00:00",
        "end_time":   "2026-05-26T14:45:00",
        "markers": [
            {
                "time_range": {"start": "2026-05-26T14:00:00", "end": "2026-05-26T14:05:00"},
                "content": "课程介绍"
            },
            {
                "time_range": [300, 600],
                "content": "第一节：基础概念"
            }
        ]
    }

POST /api/marker
    向已有会话追加单个打点（增量接收）
    请求体必须包含 "session_id" 字段

PATCH /api/session/{id}
    更新会话的 start_time / end_time

GET  /api/sessions
    列出所有已保存会话

GET  /api/session/{id}
    获取指定会话的完整记录

── 运行 ───────────────────────────────────────────────────────────────────────
    python marker_server.py [--port 8766] [--host 0.0.0.0] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from marker_store import MarkerStore

log = logging.getLogger(__name__)

_CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


# ── 请求处理器 ────────────────────────────────────────────────────────────────

class MarkerHandler:
    def __init__(self, store: MarkerStore):
        self._store = store

    # POST /api/session
    async def post_session(self, request: web.Request) -> web.Response:
        data = await _parse_json(request)
        if data is None:
            return _err(400, "请求体必须是有效 JSON")
        try:
            result = await _run(self._store.save_session, data)
            return _ok(result)
        except ValueError as e:
            return _err(400, str(e))
        except Exception:
            log.exception("post_session 异常")
            return _err(500, "服务器内部错误")

    # POST /api/marker
    async def post_marker(self, request: web.Request) -> web.Response:
        data = await _parse_json(request)
        if data is None:
            return _err(400, "请求体必须是有效 JSON")
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return _err(400, "缺少必填字段: session_id")
        try:
            result = await _run(self._store.append_marker, session_id, data)
            return _ok(result)
        except Exception:
            log.exception("post_marker 异常")
            return _err(500, "服务器内部错误")

    # PATCH /api/session/{id}
    async def patch_session(self, request: web.Request) -> web.Response:
        sid  = request.match_info["sid"]
        data = await _parse_json(request)
        if data is None:
            return _err(400, "请求体必须是有效 JSON")
        try:
            result = await _run(
                self._store.update_session_time,
                sid,
                data.get("start_time"),
                data.get("end_time"),
            )
            return _ok(result)
        except FileNotFoundError as e:
            return _err(404, str(e))
        except Exception:
            log.exception("patch_session 异常")
            return _err(500, "服务器内部错误")

    # GET /api/sessions
    async def get_sessions(self, request: web.Request) -> web.Response:
        sessions = await _run(self._store.list_sessions)
        return _ok(sessions)

    # GET /api/session/{id}
    async def get_session(self, request: web.Request) -> web.Response:
        sid    = request.match_info["sid"]
        record = await _run(self._store.get_session, sid)
        if record is None:
            return _err(404, f"会话不存在: {sid}")
        return _ok(record)

    # OPTIONS（CORS 预检）
    async def options(self, request: web.Request) -> web.Response:
        return web.Response(
            headers={
                **_CORS_HEADERS,
                "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )


# ── 应用构建 ──────────────────────────────────────────────────────────────────

def build_app(store: MarkerStore) -> web.Application:
    app     = web.Application()
    handler = MarkerHandler(store)

    app.router.add_post  ("/api/session",       handler.post_session)
    app.router.add_post  ("/api/marker",         handler.post_marker)
    app.router.add_patch ("/api/session/{sid}",  handler.patch_session)
    app.router.add_get   ("/api/sessions",       handler.get_sessions)
    app.router.add_get   ("/api/session/{sid}",  handler.get_session)
    app.router.add_route ("OPTIONS", "/{path_info:.*}", handler.options)

    return app


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

async def _parse_json(request: web.Request):
    try:
        return await request.json()
    except Exception:
        return None


async def _run(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args))


def _ok(data) -> web.Response:
    return web.Response(
        text=json.dumps({"status": "ok", "data": data}, ensure_ascii=False),
        content_type="application/json",
        headers=_CORS_HEADERS,
    )


def _err(status: int, message: str) -> web.Response:
    return web.Response(
        status=status,
        text=json.dumps({"status": "error", "message": message}, ensure_ascii=False),
        content_type="application/json",
        headers=_CORS_HEADERS,
    )


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="内容打点 HTTP 接收服务")
    parser.add_argument("--port",       type=int, default=8766,    help="监听端口（默认 8766）")
    parser.add_argument("--host",       default="0.0.0.0",         help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--output-dir", default=None,              help="数据保存目录（默认 EventMarker/marker_data/）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    store = MarkerStore(output_dir=args.output_dir)
    app   = build_app(store)

    log.info(f"内容打点服务已启动: http://{args.host}:{args.port}")
    log.info(f"  POST /api/session    — 接收完整会话")
    log.info(f"  POST /api/marker     — 追加单个打点")
    log.info(f"  GET  /api/sessions   — 列出所有会话")
    log.info(f"  GET  /api/session/{{id}} — 查看指定会话")
    log.info(f"数据保存目录: {store._root.resolve()}")

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
