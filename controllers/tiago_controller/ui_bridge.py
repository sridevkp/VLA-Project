"""Put this file in the same folder as your TIAGo controller (next to tiago_pick_place.py).

Streams the RGB camera to the robot window as "img:<base64 jpeg>" and chat replies as
"text:<reply>". Messages typed in the window are queued in `bridge.commands`.
Look for lines starting with [ui] in the Webots console.
"""
import base64
import os
import tempfile
import time


class UIBridge:
    def __init__(self, robot, camera, period_ms=250, quality=60):
        self.robot = robot
        self.camera = camera
        self.period_ms = period_ms          # time between camera frames (250 ms = 4 fps)
        self.quality = quality              # JPEG quality 1-100 (lower = smaller messages)
        self.commands = []                  # texts received from the window, oldest first
        self._path = os.path.join(tempfile.gettempdir(), "tiago_cam_frame.jpg")
        self._next_frame_s = 0.0          # real (wall-clock) time, not simulation time
        self._busy = False                  # guards against re-entrancy
        self._ticks = 0
        self._frames_sent = 0
        self._save_fail = 0
        self._got_msgs = 0

        print("[ui] bridge created")
        print("[ui] wwiSendText available:", hasattr(robot, "wwiSendText"),
              "| wwiReceiveText available:", hasattr(robot, "wwiReceiveText"))
        print("[ui] camera:", camera.getName() if camera else None,
              "size:", (camera.getWidth(), camera.getHeight()) if camera else None)
        print("[ui] temp frame file:", self._path)

    # ---- controller -> window ----
    def say(self, text):
        self.robot.wwiSendText("text:" + str(text))

    def _send_frame(self):
        # saveImage returns 0 on success; it fails until the camera has produced a frame
        rc = self.camera.saveImage(self._path, self.quality)
        if rc != 0:
            self._save_fail += 1
            if self._save_fail in (1, 20, 200):
                print(f"[ui] camera.saveImage failed (code {rc}), count={self._save_fail}")
            return
        with open(self._path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        self.robot.wwiSendText("img:" + b64)
        self._frames_sent += 1
        if self._frames_sent in (1, 10, 100):
            print(f"[ui] frame #{self._frames_sent} sent ({len(b64) // 1024} KB base64)")

    # ---- window -> controller ----
    def _poll(self):
        while True:
            msg = self.robot.wwiReceiveText()
            if not msg:
                break
            msg = msg.strip()
            self._got_msgs += 1
            if self._got_msgs <= 5:
                print(f"[ui] received from window: {msg!r}")
            if msg == "hello":              # the window announces itself when it loads
                self.say("TIAGo controller connected.")
            else:
                self.commands.append(msg)
                self.say(f"Received: {msg}")

    # ---- call once per simulation step ----
    def tick(self):
        if self._busy:
            return
        self._busy = True
        try:
            self._ticks += 1
            if self._ticks == 1:
                print("[ui] first tick: the step wrapper is working")
            self._poll()
            now = time.monotonic()           # wall clock: frame rate stays fixed even at fast sim speed
            if now >= self._next_frame_s:
                self._next_frame_s = now + self.period_ms / 1000.0
                self._send_frame()
        except Exception as e:              # never let the UI crash the pick-and-place
            print("[ui] error:", repr(e))
        finally:
            self._busy = False