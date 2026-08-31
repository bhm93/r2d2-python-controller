# Fanhome / DeAgostini R2-D2 Python Controller

A custom Python desktop application to control the **DeAgostini / Fanhome "Build Your Own R2-D2"** scale model over local Wi-Fi, using the same local WebSocket API as the official mobile app.

Drive the robot, stream its live camera, trigger sounds/lights/routines, and even let it recognize hand gestures or follow you around, all from your desktop.

## 🚀 Features

- **Full manual control**: directional pad, dome/head rotation, arm, lightsaber, projectors, LCD animations, patrol and dance routines, mute toggle, sound board, and a "restore normal lights" shortcut.
- **Adjustable drive speed**: a slider controls the motor power used by the directional pad and keyboard driving.
- **Keyboard controls**: arrow keys drive the wheels, `Z`/`X` rotate the head — in addition to the on-screen buttons. Both mouse and keyboard use hold-to-move (press and hold to keep moving, release to stop).
- **Live camera feed** (`ws://<robot-ip>:12121`), correctly rotated for display.
- **Computer vision modes**, built on MediaPipe, running on the live camera feed:
  - **Hand gesture recognition**: closed fist (stop), open palm (greeting sound), thumbs up (dance), victory sign (lightsaber).
  - **"Follow me"**: detects the nearest person by body silhouette (works even from behind, unlike a face detector) and drives the robot to keep them centered and at a comfortable distance.
- **Battery level monitoring**, read from the robot's periodic status broadcasts.
- **Resilient networking**: automatic reconnect handling, retry-with-backoff if the robot's camera socket is still occupied by a stale connection, and a safety watchdog that stops the robot if the video feed drops out mid-movement.

## 🛠️ Requirements

- Python 3.9+ (uses `asyncio.to_thread`)
- Windows, macOS, or Linux

### Installation

```bash
git clone https://github.com/bhm93/r2d2-python-controller.git
cd r2d2-python-controller
pip install -r requirements.txt
```

The MediaPipe models used for gesture/person detection (a few MB each) are downloaded automatically on first run and cached alongside the script.

## 📡 How to Connect to R2-D2

1. Turn on your R2-D2 robot in AP (Access Point) mode.
2. Connect your computer's Wi-Fi to R2-D2's own Wi-Fi network.
3. Verify your PC gets an IP in the `192.168.43.x` range (the robot itself is at `192.168.43.1`).

## 💻 Usage

```bash
python r2_control_gui.py
```

Click **"Conectar control"** to authenticate, then **"Iniciar cámara"** to start receiving live video. Pick a vision mode ("Gestos con la mano" / "Sígueme") from the dropdown to enable gesture recognition or person-following.

**Driving:**
- Mouse: hold ▲◀▶▼ to move, release to stop.
- Keyboard: hold the arrow keys to move, `Z`/`X` to rotate the head.
- Speed slider controls drive power (10–100).

## 🔌 Protocol Technical Breakdown

### Network Architecture

- **Control port** (JSON over WebSocket): `ws://192.168.43.1:8887`
- **Camera stream** (binary JPEG frames): `ws://192.168.43.1:12121`

### Connection Sequence & Keep-Alive

1. Handshake: `{"cmd": "grantAccess", "uuid": "<UUID>", "device_name": "PC-R2D2", "seq": 1}`
2. Enable control: `{"cmd": "user_control", "enable": true}`
3. **Keep-alive**: the robot's control socket drops the session after ~12 seconds without a fresh `user_control: true`, so it must be re-sent periodically (this app refreshes every 8s).
4. **Motor watchdog**: the drive motor board has its own short-lived watchdog — a single `move`/`head-dir` command is not enough to sustain continuous motion. The official app resends the active `move`/`head-dir` command roughly every 300ms while a direction is held; this app does the same, and also periodically sends `reset-wdt` to the motor board.

### Video Stream

Frames received on port `12121` are raw JPEG binary images, one client at a time (the robot rejects a second simultaneous camera connection). Frames arrive rotated relative to the on-screen orientation; rotate 90° clockwise before rendering.

## ⚠️ Safety Disclaimer

This project talks to your robot over its local, undocumented protocol — not the official app. Bugs, dropped connections, or timing issues could cause unexpected or sustained movement. **Supervise the robot while it's connected, keep it away from stairs/edges/pets/people, and be ready to power it off.** Use at your own risk; see [LICENSE](LICENSE) for the full disclaimer.

## 📄 License

[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — free to use, modify, and share for personal, educational, or other non-commercial purposes, with no warranty. See [LICENSE](LICENSE) for the full terms, plus a project-specific hardware/safety disclaimer.
