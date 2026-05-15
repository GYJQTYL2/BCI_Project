import asyncio
from bleak import BleakClient
import sys
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import muse_athena_protocol as proto

ADDRESS  = "220594BA-88EC-8B79-7BF0-B48BB54AAF16"
CTRL     = proto.CONTROL_UUID
SENSOR   = proto.SENSOR_UUID

count = 0

def on_sensor(char, data: bytearray):
    global count
    count += 1
    if count <= 5:
        print(f"[SENSOR] #{count}  len={len(data)}  hex={data.hex()[:40]}...")

def on_ctrl(char, data: bytearray):
    text = data.decode('ascii', errors='replace').strip('\x00').strip()
    if text:
        print(f"[CTRL] {text}")

async def main():
    print(f"连接 {ADDRESS} ...")
    async with BleakClient(ADDRESS) as client:
        print("已连接，执行 Athena 初始化序列...\n")
        await client.start_notify(CTRL,   on_ctrl)
        await client.start_notify(SENSOR, on_sensor)

        for desc, cmd, delay in proto.get_init_sequence("p21"):
            print(f"  >> {desc}: {cmd.hex()}")
            await client.write_gatt_char(CTRL, cmd, response=False)
            await asyncio.sleep(delay)

        print("\n等待 5 秒数据...\n")
        await asyncio.sleep(5)
        print(f"\n收到 {count} 个传感器包")
        if count > 0:
            print("✓ 连接成功！设备正在流式传输数据。")
        else:
            print("✗ 仍无数据。")

asyncio.run(main())
