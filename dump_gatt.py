import asyncio
from bleak import BleakClient

ADDRESS = "220594BA-88EC-8B79-7BF0-B48BB54AAF16"

async def dump():
    print(f"连接 {ADDRESS} ...")
    async with BleakClient(ADDRESS) as client:
        print(f"已连接: {client.is_connected}\n")
        for service in client.services:
            print(f"Service: {service.uuid}  ({service.description})")
            for char in service.characteristics:
                print(f"  Char: {char.uuid}  props={char.properties}  handle={char.handle}")

asyncio.run(dump())
