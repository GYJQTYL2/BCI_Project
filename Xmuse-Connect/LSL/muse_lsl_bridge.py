"""
MuseS Athena → LSL Bridge

Connects to a MuseS (firmware 3.x, protocol v4 / Athena) device via BLE
and publishes sensor data as Lab Streaming Layer (LSL) outlets so that
LSL-singal_device.py (or any other LSL consumer) can discover them.

Usage:
    python3 muse_lsl_bridge.py
    python3 muse_lsl_bridge.py --address 220594BA-88EC-8B79-7BF0-B48BB54AAF16
    python3 muse_lsl_bridge.py --preset p1034   # full sensors (EEG + IMU + PPG)
    python3 muse_lsl_bridge.py --preset p21     # EEG only (default)

Streams created:
    EEG      – 4 ch (TP9 AF7 AF8 TP10) at 256 Hz
    IMU      – 6 ch (AccX AccY AccZ GyrX GyrY GyrZ) at 52 Hz
    PPG      – 4 ch (PPG optics) at 64 Hz   [p1034 only]
    Battery  – 3 ch (Percent% Voltage_V Temp_C) irregular rate (from control channel)
"""

import asyncio
import argparse
import json
import sys
import time
import logging
from typing import Optional

import numpy as np
from bleak import BleakClient, BleakScanner

sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import muse_athena_protocol as proto
import pylsl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LSL outlet factory
# ---------------------------------------------------------------------------

def _make_outlet(name: str, stream_type: str, channels: list[str],
                 srate: float, fmt=pylsl.cf_float32) -> pylsl.StreamOutlet:
    info = pylsl.StreamInfo(
        name=name,
        type=stream_type,
        channel_count=len(channels),
        nominal_srate=srate,
        channel_format=fmt,
        source_id=f"MuseS_{name}",
    )
    ch_xml = info.desc().append_child("channels")
    for ch in channels:
        ch_xml.append_child("channel").append_child_value("label", ch)
    outlet = pylsl.StreamOutlet(info)
    log.info(f"LSL outlet created: {name} ({stream_type}) "
             f"{len(channels)}ch @ {srate}Hz")
    return outlet


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------

class MuseLSLBridge:
    def __init__(self, address: str, preset: str = "p21"):
        self.address = address
        self.preset  = preset
        self._outlets: dict = {}
        self._packet_count = 0
        self._sample_counts: dict = {}
        self._running = False
        self._ctrl_buf = ""      # 跨包 JSON 拼接缓冲
        self._last_ps: Optional[int] = None  # 上一次的 ps 值


    # -- LSL outlets --------------------------------------------------------

    def _ensure_eeg_outlet(self, n_channels: int):
        key = f"EEG{n_channels}"
        if key not in self._outlets:
            ch = proto.EEG_CHANNELS_4 if n_channels == 4 else proto.EEG_CHANNELS_8
            self._outlets[key] = _make_outlet("MuseS-EEG", "EEG", ch, 256.0)
        return self._outlets[key]

    def _ensure_imu_outlet(self):
        if "IMU" not in self._outlets:
            self._outlets["IMU"] = _make_outlet(
                "MuseS-IMU", "IMU",
                ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"], 52.0)
        return self._outlets["IMU"]

    def _ensure_ppg_outlet(self, n_channels: int):
        key = f"PPG{n_channels}"
        if key not in self._outlets:
            chs = [f"PPG_{i+1}" for i in range(n_channels)]
            self._outlets[key] = _make_outlet("MuseS-PPG", "PPG", chs, 64.0)
        return self._outlets[key]

    def _ensure_battery_outlet(self):
        if "Battery" not in self._outlets:
            self._outlets["Battery"] = _make_outlet(
                "MuseS-Battery", "Battery",
                ["Percent%", "Voltage_V", "Temp_C"], 0.0)
        return self._outlets["Battery"]

    # -- Packet handler -----------------------------------------------------

    def _on_sensor(self, _char, data: bytearray):
        self._packet_count += 1
        ts = pylsl.local_clock()

        try:
            parsed = proto.parse_payload(bytes(data))
        except Exception as e:
            log.info(f"Parse error: {e}")
            return

        # EEG
        for sub in parsed.get("EEG", []):
            samples: np.ndarray = sub["data"]   # shape (n_samples, n_channels)
            outlet = self._ensure_eeg_outlet(sub["n_channels"])
            srate  = sub["sample_rate"]
            n      = sub["n_samples"]
            for i, row in enumerate(samples):
                sample_ts = ts - (n - 1 - i) / srate
                outlet.push_sample(row.tolist(), sample_ts)
            self._sample_counts["EEG"] = self._sample_counts.get("EEG", 0) + n

        # IMU (acc + gyro)
        for sub in parsed.get("ACCGYRO", []):
            samples: np.ndarray = sub["data"]   # shape (n_samples, 6)
            outlet = self._ensure_imu_outlet()
            srate  = sub["sample_rate"]
            n      = sub["n_samples"]
            for i, row in enumerate(samples):
                sample_ts = ts - (n - 1 - i) / srate
                outlet.push_sample(row.tolist(), sample_ts)
            self._sample_counts["IMU"] = self._sample_counts.get("IMU", 0) + n

        # PPG / Optics
        for sub in parsed.get("OPTICS", []):
            samples: np.ndarray = sub["data"]
            outlet = self._ensure_ppg_outlet(sub["n_channels"])
            srate  = sub["sample_rate"]
            n      = sub["n_samples"]
            for i, row in enumerate(samples):
                sample_ts = ts - (n - 1 - i) / srate
                outlet.push_sample(row.tolist(), sample_ts)
            self._sample_counts["PPG"] = self._sample_counts.get("PPG", 0) + n

        # Status log every 256 packets (~5 s)
        if self._packet_count % 256 == 0:
            parts = [f"{k}={v}" for k, v in self._sample_counts.items()]
            log.info(f"Streaming – packets={self._packet_count}  samples: {', '.join(parts)}")

    def _on_ctrl(self, _char, data: bytearray):
        try:
            # 第 0 字节是长度前缀，从第 1 字节开始才是 JSON 内容；去掉填充的 \x00
            self._ctrl_buf += data[1:].decode("ascii", errors="replace").replace("\x00", "")

            while "}" in self._ctrl_buf:
                end = self._ctrl_buf.index("}") + 1
                fragment = self._ctrl_buf[:end].strip()

                # 跳过分隔符（0、逗号、空白），定位到下一个 { 开头
                rest = self._ctrl_buf[end:]
                next_brace = rest.find("{")
                self._ctrl_buf = rest[next_brace:] if next_brace >= 0 else ""

                try:
                    msg = json.loads(fragment)
                except json.JSONDecodeError:
                    log.debug(f"[CTRL] 无法解析片段: {fragment!r}")
                    continue

                # 固件信息（启动时一次）
                if "fw" in msg:
                    log.info(f"[CTRL] 固件: fw={msg['fw']}  hw={msg.get('hw')}  pv={msg.get('pv')}")

                # 电池 + 状态包 → Battery LSL outlet
                if "bp" in msg:
                    bp  = float(msg["bp"])
                    tp  = float(msg.get("tp", 0.0))   # temperature (°C) if present
                    log.info(f"[CTRL] 电池: {bp:.1f}%  tp={tp}  ps={msg.get('ps')}  ln={msg.get('ln')}")
                    self._ensure_battery_outlet().push_sample(
                        [bp, 0.0, tp], pylsl.local_clock()
                    )

                # 记录 ps 变化（设备状态机，非接触质量，仅供调试）
                if "ps" in msg:
                    ps = int(msg["ps"])
                    if self._last_ps is not None and ps != self._last_ps:
                        log.info(
                            f"[CTRL] ps 变化: {self._last_ps} → {ps}  "
                            f"(diff bits: {self._last_ps ^ ps:#06x})"
                        )
                    self._last_ps = ps

                # 仅记录真正未知的字段（排除已知设备信息字段）
                known_keys = {
                    "fw", "hw", "pv", "bp", "ps", "rc", "ln", "tp",
                    # 设备标识字段（启动时出现一次，固定内容）
                    "bn", "ap", "sp", "hb", "bl", "be",
                    "hn", "sn", "ma", "hs", "id",
                }
                unknown = {k: v for k, v in msg.items() if k not in known_keys}
                if unknown:
                    log.info(f"[CTRL] 未知字段: {unknown}")

        except Exception:
            log.exception("[CTRL] 回调异常")

    # -- Main connect loop --------------------------------------------------

    async def run(self):
        self._running = True
        log.info(f"Connecting to {self.address} ...")

        async with BleakClient(self.address) as client:
            log.info("Connected. Running Athena init sequence...")

            await client.start_notify(proto.CONTROL_UUID,  self._on_ctrl)
            await client.start_notify(proto.SENSOR_UUID,   self._on_sensor)

            for desc, cmd, delay in proto.get_init_sequence(self.preset):
                await client.write_gatt_char(proto.CONTROL_UUID, cmd, response=False)
                await asyncio.sleep(delay)

            log.info("Init complete – LSL streams are live. Press Ctrl+C to stop.")

            _status_interval = 2.0  # 每 2 秒轮询一次设备状态
            _last_status_t   = 0.0
            while self._running and client.is_connected:
                await asyncio.sleep(0.2)
                now = asyncio.get_event_loop().time()
                if now - _last_status_t >= _status_interval:
                    await client.write_gatt_char(
                        proto.CONTROL_UUID, proto.COMMANDS["s"], response=False
                    )
                    _last_status_t = now

        log.info("Disconnected.")

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _find_muse(retries: int = 3) -> Optional[str]:
    for attempt in range(1, retries + 1):
        log.info(f"扫描 Muse 设备（第 {attempt}/{retries} 次，10 秒）...")
        devices = await BleakScanner.discover(timeout=10.0)
        for d in devices:
            if d.name and "muse" in d.name.lower():
                log.info(f"发现设备: {d.name}  {d.address}")
                return d.address
        if attempt < retries:
            log.warning("未发现 Muse 设备，3 秒后重试。请确认设备已开机并处于广播状态（LED 持续闪烁）。")
            await asyncio.sleep(3)
    return None


async def _main(args):
    address = args.address
    if not address:
        address = await _find_muse(retries=3)
        if not address:
            log.error("未找到 Muse 设备。请长按电源键直到听到两声提示音后重试。")
            sys.exit(1)

    bridge = MuseLSLBridge(address=address, preset=args.preset)

    try:
        await bridge.run()
    except KeyboardInterrupt:
        bridge.stop()
        log.info("Bridge stopped.")


def main():
    parser = argparse.ArgumentParser(description="MuseS Athena → LSL bridge")
    parser.add_argument("--address", default="",
                        help="BLE address/UUID of the MuseS device (auto-scan if omitted)")
    parser.add_argument("--preset", default="p21",
                        choices=["p21", "p1034", "p1035"],
                        help="Sensor preset: p21=EEG only, p1034=EEG+IMU+PPG (default: p21)")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
