import asyncio
from bleak import BleakScanner

async def scan():
    print("扫描 BLE 设备（10秒）...")
    devices = await BleakScanner.discover(timeout=10.0)
    if not devices:
        print("未发现任何 BLE 设备")
        print("\n可能原因：")
        print("  1. 终端没有蓝牙权限：系统设置 → 隐私与安全性 → 蓝牙 → 允许 Terminal")
        print("  2. 设备未进入广播状态：长按电源键直到听到两声提示音")
        return
    print(f"\n发现 {len(devices)} 个设备：")
    muse_devices = [d for d in devices if d.name and 'muse' in d.name.lower()]
    if muse_devices:
        print("\n*** 发现 Muse 设备：***")
        for d in muse_devices:
            print(f"  名称: {d.name}   地址: {d.address}")
    else:
        print("\n未找到 Muse 设备，所有设备列表：")
        for d in sorted(devices, key=lambda x: x.name or ""):
            name = d.name or "(无名称)"
            print(f"  {name:<35}  {d.address}")

asyncio.run(scan())
