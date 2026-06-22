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


def _sanitize(obj):
    """递归将 NaN / Inf 替换为 None，避免 JSON 序列化失败"""
    import math
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class WebSocketServer:
    def __init__(
        self,
        bridge,
        port: int = 8765,
        push_hz: float = 10.0,
        history_api=None,       # HistoryAPI 实例（可选），注入历史数据路由
        processor=None,         # RealTimeEEGProcessor 实例（可选），用于基线录制
    ):
        self._bridge = bridge
        self._port = port
        self._interval = 1.0 / push_hz
        self._history_api = history_api
        self._processor = processor
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def set_processor(self, processor) -> None:
        """绑定 RealTimeEEGProcessor，可在 start() 后调用"""
        self._processor = processor

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

        async def handle_baseline_start(request):
            if self._processor is None:
                return web.Response(status=503, text="processor not available")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._processor.start_baseline)
            self._bridge.set_baseline_status("recording")

            # 30 秒后自动停止
            async def _auto_stop():
                await asyncio.sleep(30)
                ok = await loop.run_in_executor(None, self._processor.stop_baseline)
                self._bridge.set_baseline_status("ready" if ok else "idle")

            asyncio.create_task(_auto_stop())
            return web.Response(text="ok")

        async def handle_baseline_stop(request):
            if self._processor is None:
                return web.Response(status=503, text="processor not available")
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(None, self._processor.stop_baseline)
            self._bridge.set_baseline_status("ready" if ok else "idle")
            return web.json_response({"ok": ok})

        app = web.Application()
        app.router.add_get("/", handle_index)
        app.router.add_get("/favicon.ico", handle_favicon)
        app.router.add_get("/ws", handle_ws)
        app.router.add_post("/baseline/start", handle_baseline_start)
        app.router.add_post("/baseline/stop", handle_baseline_stop)
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
            try:
                data = json.dumps(self._bridge.snapshot(), allow_nan=False)
            except (ValueError, TypeError):
                data = json.dumps(_sanitize(self._bridge.snapshot()), allow_nan=False)
            dead = set()
            for ws in list(clients):
                try:
                    await ws.send_str(data)
                except Exception:
                    dead.add(ws)
            clients -= dead
