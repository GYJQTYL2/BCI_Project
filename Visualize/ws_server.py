"""
aiohttp WebSocket + HTTP 服务

提供：
  GET  /           → static/index.html
  GET  /favicon.ico → 204 No Content
  WS   /ws         → 每 100ms 推送 DataBridge.snapshot() 的 JSON

在 daemon 线程中运行，不阻塞主采集循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


class WebSocketServer:
    def __init__(
        self,
        bridge,
        port: int = 8765,
        push_hz: float = 10.0,
        history_api=None,       # HistoryAPI 实例（可选），注入历史数据路由
    ):
        self._bridge = bridge
        self._port = port
        self._interval = 1.0 / push_hz
        self._history_api = history_api
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import aiohttp  # noqa: F401 — 提前检测，给出明确错误
        except ImportError:
            print("[ws_server] ERROR: aiohttp 未安装，请运行: pip install aiohttp")
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            print(f"[ws_server] ERROR: 服务异常退出 → {exc}")
            log.exception("WebSocket 服务异常")

    async def _serve(self) -> None:
        from aiohttp import web

        self._stop_event = asyncio.Event()
        clients: set = set()

        async def handle_index(request):
            return web.FileResponse(_INDEX_HTML)

        async def handle_static(request):
            filename = request.match_info["filename"]
            filepath = _STATIC_DIR / filename
            if filepath.exists() and filepath.is_file():
                return web.FileResponse(filepath)
            return web.Response(status=404)

        async def handle_favicon(request):
            return web.Response(status=204)

        async def handle_ws(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            clients.add(ws)
            print(f"[ws_server] 客户端已连接，当前连接数: {len(clients)}")
            try:
                async for _ in ws:
                    pass
            finally:
                clients.discard(ws)
                print(f"[ws_server] 客户端已断开，当前连接数: {len(clients)}")
            return ws

        app = web.Application()
        app.router.add_get("/", handle_index)
        app.router.add_get("/favicon.ico", handle_favicon)
        app.router.add_get("/ws", handle_ws)
        if self._history_api is not None:
            self._history_api.register_routes(app)
        app.router.add_get("/{filename}", handle_static)

        asyncio.create_task(self._push_loop(clients))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        await site.start()

        print(f"[ws_server] 服务已监听 http://0.0.0.0:{self._port}")
        await self._stop_event.wait()
        await runner.cleanup()

    async def _push_loop(self, clients: set) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if not clients:
                continue
            data = json.dumps(self._bridge.snapshot())
            dead = set()
            for ws in list(clients):
                try:
                    await ws.send_str(data)
                except Exception:
                    dead.add(ws)
            clients -= dead
