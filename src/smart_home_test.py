import asyncio
from kasa import Discover, Module
import logging

async def test():
    dev = await Discover.discover_single("192.168.1.195",
                                         username="danieljhkoo@gmail.com", password="Djhk0418!")
    await dev.update()
    light = dev.modules[Module.Light]
    await dev.turn_on()
    await light.set_brightness(30)
    print(dev.alias, light.valid_temperature_range)

asyncio.run(test())