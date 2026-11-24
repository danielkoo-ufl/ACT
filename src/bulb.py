import asyncio, threading, time
from kasa import Discover, Module



class BulbController:
    # host = IP address of the bulb
    # username = username for the bulb
    # password = password for the bulb
    # scenes = dictionary mapping event names to scene configurations
    # min_conf = minimum confidence threshold for applying a scene
    # hold = number of consecutive predictions required to apply a scene
    def __init__(self, host, username, password, scenes,
                 min_conf=0.5, hold=1, cooldown=10):
        self.scenes = scenes
        self.host, self.user, self.pw = host, username, password
        self.min_conf, self.hold, self.cooldown = min_conf, hold, cooldown
        self.dev = None
        self.current = None            # scene currently applied
        self.candidate, self.streak = None, 0
        self.last_change = 0
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._connect(), self.loop)

    async def _connect(self):
        try:
            self.dev = await Discover.discover_single(
                self.host, username=self.user, password=self.pw)
            await self.dev.update()
            print(f"[bulb] connected: {self.dev.alias}")
        except Exception as e:
            print(f"[bulb] connect failed: {e}")

    async def _apply(self, event):
        if self.dev is None:
            return
        try:
            s = self.scenes.get(event)
            if not s:
                return
            await self.dev.update()
            await self.dev.turn_on()
            light = self.dev.modules[Module.Light]
            await light.set_brightness(s["brightness"])
            if light.is_variable_color_temp:
                lo, hi = light.valid_temperature_range
                await light.set_color_temp(max(lo, min(hi, s["color_temp"])))
            print(f"[bulb] scene -> {event}")
        except Exception as e:
            print(f"[bulb] error: {e}")

    def update(self, event, conf):
        """Call once per prediction from the main loop (non-blocking)."""
        if event not in self.scenes or conf < self.min_conf:
            self.candidate, self.streak = None, 0
            return
        if event == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = event, 1

        if (self.streak >= self.hold and event != self.current
                and time.time() - self.last_change > self.cooldown):
            self.current, self.last_change = event, time.time()
            asyncio.run_coroutine_threadsafe(self._apply(event), self.loop)