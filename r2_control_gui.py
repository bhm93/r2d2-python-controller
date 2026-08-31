"""Panel gráfico local para el R2-D2 DeAgostini, con Visión Artificial.

Requiere:
    pip install websockets pillow opencv-python mediapipe numpy

Ejecuta:
    python r2_control_gui.py

Dos modos de visión (menú "Modo Visión" de la barra de herramientas):
  - "Gestos con la mano": reconoce puño cerrado, palma abierta, pulgar arriba
    y seña de victoria con MediaPipe GestureRecognizer (landmarks reales de
    la mano, no diferencia de movimiento) y dispara una acción del robot.
  - "Sígueme": detecta la silueta de la persona más prominente con MediaPipe
    ObjectDetector (categoría "person") y gira/avanza el robot para
    mantenerla centrada y a media distancia. Al trackear el cuerpo completo
    en vez de solo la cara, sigue funcionando aunque la persona esté de
    espaldas o de perfil.

Los modelos de MediaPipe (.task / .tflite) se descargan automáticamente la
primera vez, junto al script.
"""

import asyncio
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import websockets
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image, ImageTk


ROBOT_IP = "192.168.43.1"
CONTROL_URI = f"ws://{ROBOT_IP}:8887"
VIDEO_URI = f"ws://{ROBOT_IP}:12121"
CLIENT_UUID = "663f920a-a33c-4e3a-93fb-f40af97be727"

# Ángulos de giro "move" que hacen que el robot gire realmente HACIA el lado
# de la imagen citado (validado en el robot real con el modo Sígueme:
# objetivo a la izquierda de la imagen -> angle=-90, igual que el botón "◀").
TURN_TOWARD_IMAGE_LEFT = -90
TURN_TOWARD_IMAGE_RIGHT = 90

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
PERSON_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
)
GESTURE_MODEL_PATH = "gesture_recognizer.task"
PERSON_MODEL_PATH = "person_detector.tflite"

# Gesto MediaPipe -> acción del robot. Las categorías posibles del modelo son
# ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down",
#  "Thumb_Up", "Victory", "ILoveYou"].
GESTURE_ACTIONS = {
    "Closed_Fist": ("stop", None, "Puño -> Parar"),
    "Open_Palm": ("sound", 3, "Palma abierta -> Saludo"),
    "Thumb_Up": ("mode", 10, "Pulgar arriba -> Baile"),
    "Victory": ("mode", 6, "Victoria -> Sable"),
}
GESTURE_MIN_SCORE = 0.65
GESTURE_COOLDOWN = 1.5

# Umbrales del modo "Sígueme" (proporción del área de la silueta de la
# persona sobre el frame). Son un punto de partida razonable, pero conviene
# afinarlos viendo el vídeo real: si avanza/retrocede demasiado pronto o
# tarde, ajusta estos dos valores.
FOLLOW_DEADZONE_RATIO = 0.15
# FOLLOW_FAR_AREA_RATIO alto a propósito: el robot debe empezar a avanzar
# mientras la persona todavía es fácil de detectar con confianza, no esperar
# a que se haga tan pequeña en el encuadre que el detector deje de verla.
FOLLOW_FAR_AREA_RATIO = 0.30    # silueta pequeña -> persona lejos -> avanzar
FOLLOW_NEAR_AREA_RATIO = 0.45   # silueta grande -> persona cerca -> retroceder
FOLLOW_LOST_TIMEOUT = 1.5       # sin detección durante esto -> parar y esperar

# Avanzar/retroceder en "Sígueme" es continuo: se manda un único "move" que
# se mantiene mientras el estado deseado no cambie (no hay riesgo real de
# "pasarse" de la distancia buena en el intervalo entre un frame y otro).
# Girar es distinto: si se hace continuo, el robot sigue girando entre un
# frame y el siguiente (~100 ms) y se pasa del centro, con lo que el
# siguiente frame ve a la persona ya al otro lado y corrige en sentido
# contrario -> vaivén izquierda-derecha sin parar. Por eso el giro va a
# impulsos cortos y con su propio intervalo mínimo entre correcciones.
FOLLOW_TURN_POWER = 35
FOLLOW_FORWARD_POWER = 45
FOLLOW_BACKWARD_POWER = 40
FOLLOW_TURN_BURST_DURATION = 0.15   # cuánto dura cada impulso de giro
FOLLOW_TURN_BURST_COOLDOWN = 0.3    # tiempo mínimo entre impulsos de giro sucesivos
FOLLOW_MIN_COMMAND_INTERVAL = 0.05  # anti-flood si el estado oscila entre frames

# Salvaguarda: si "Sígueme" está en medio de un movimiento continuo y dejan
# de llegar fotogramas de la cámara (WiFi cortado, robot colgado, etc.)
# durante más de esto, se manda una parada de emergencia aunque nadie la pida.
FOLLOW_WATCHDOG_INTERVAL = 0.5
FOLLOW_WATCHDOG_STALE_FRAMES = 1.0


def ensure_model_file(local_path, url, human_name):
    """Descarga un modelo de MediaPipe si no existe ya en disco."""
    if os.path.exists(local_path):
        return local_path
    try:
        print(f"Descargando modelo de {human_name}...")
        urllib.request.urlretrieve(url, local_path)
        print("Descarga completada con éxito.")
        return local_path
    except Exception as e:
        print(f"Error al descargar el modelo de {human_name}: {e}")
        return None


class VisionProcessor:
    """Analiza fotogramas con MediaPipe: gestos de mano y seguimiento de personas."""

    def __init__(self):
        self.gesture_recognizer = self._load_gesture_recognizer()
        self.person_detector = self._load_person_detector()

        self._frame_idx = 0
        self.last_gesture_name = None
        self.last_gesture_time = 0.0

        # Estado deseado actual del "Sígueme": None (sin objetivo aún) o uno
        # de "left"/"right"/"forward"/"backward"/"stop". Solo se manda un
        # comando nuevo al robot cuando este valor cambia.
        self.follow_state = None
        self.last_follow_time = 0.0
        self.last_face_seen_time = 0.0

    def _load_gesture_recognizer(self):
        path = ensure_model_file(GESTURE_MODEL_PATH, GESTURE_MODEL_URL, "reconocimiento de gestos")
        if not path:
            return None
        options = mp_vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        return mp_vision.GestureRecognizer.create_from_options(options)

    def _load_person_detector(self):
        """Detector de personas por silueta completa (EfficientDet-Lite/COCO),
        filtrado a la categoría "person": sigue detectando a alguien de perfil
        o de espaldas, a diferencia de un detector facial."""
        path = ensure_model_file(PERSON_MODEL_PATH, PERSON_MODEL_URL, "detección de personas")
        if not path:
            return None
        options = mp_vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=mp_vision.RunningMode.VIDEO,
            max_results=5,
            score_threshold=0.5,
            category_allowlist=["person"],
        )
        return mp_vision.ObjectDetector.create_from_options(options)

    def process_frame(self, cv_img, vision_mode):
        """Analiza la imagen según el modo activo y devuelve la imagen anotada junto al comando."""
        self._frame_idx += 1
        command = None

        if vision_mode == "Gestos con la mano" and self.gesture_recognizer:
            command = self._process_gestures(cv_img)
        elif vision_mode == "Sígueme" and self.person_detector:
            command = self._process_follow(cv_img)

        return cv_img, command

    def _process_gestures(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.gesture_recognizer.recognize_for_video(mp_img, self._frame_idx)

        command = None
        now = time.time()

        if result.gestures and result.hand_landmarks:
            top = result.gestures[0][0]
            name, score = top.category_name, top.score

            h, w = cv_img.shape[:2]
            xs = [lm.x * w for lm in result.hand_landmarks[0]]
            ys = [lm.y * h for lm in result.hand_landmarks[0]]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(cv_img, f"{name} ({score:.2f})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            action = GESTURE_ACTIONS.get(name)
            if action and score >= GESTURE_MIN_SCORE:
                changed = name != self.last_gesture_name
                cooled_down = (now - self.last_gesture_time) > GESTURE_COOLDOWN
                if changed or cooled_down:
                    command = action[:2]
                    cv2.putText(cv_img, action[2], (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    self.last_gesture_time = now
                self.last_gesture_name = name
            else:
                self.last_gesture_name = name
        else:
            self.last_gesture_name = None

        return command

    _FOLLOW_LABELS = {
        "left": ("GIRANDO IZQUIERDA", (255, 255, 0)),
        "right": ("GIRANDO DERECHA", (255, 255, 0)),
        "forward": ("AVANZANDO", (0, 255, 0)),
        "backward": ("RETROCEDIENDO", (0, 165, 255)),
        "stop": ("A DISTANCIA OK", (0, 255, 0)),
    }

    def _process_follow(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.person_detector.detect_for_video(mp_img, self._frame_idx)

        now = time.time()
        h, w = cv_img.shape[:2]
        desired_state = None

        if result.detections:
            self.last_face_seen_time = now
            det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bb = det.bounding_box
            score = det.categories[0].score if det.categories else 0.0
            x1, y1 = bb.origin_x, bb.origin_y
            x2, y2 = x1 + bb.width, y1 + bb.height
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"Persona ({score:.2f})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            center_x = x1 + bb.width / 2
            area_ratio = (bb.width * bb.height) / (w * h)
            deadzone = w * FOLLOW_DEADZONE_RATIO

            if center_x < w / 2 - deadzone:
                desired_state = "left"
            elif center_x > w / 2 + deadzone:
                desired_state = "right"
            elif area_ratio < FOLLOW_FAR_AREA_RATIO:
                desired_state = "forward"
            elif area_ratio > FOLLOW_NEAR_AREA_RATIO:
                desired_state = "backward"
            else:
                desired_state = "stop"

            label, color = self._FOLLOW_LABELS[desired_state]
            cv2.putText(cv_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(cv_img, "Sin objetivo", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if now - self.last_face_seen_time > FOLLOW_LOST_TIMEOUT:
                desired_state = "stop"
            else:
                # Fallo de detección puntual (parpadeo de un frame): mantener
                # el último estado un instante en vez de frenar en seco.
                desired_state = self.follow_state

        command = None
        if desired_state is None:
            return None

        if desired_state in ("left", "right"):
            # Giro a impulsos cortos, reevaluado con su propio cooldown
            # (más corto que el de cambio de estado, porque aquí sí
            # queremos reintentar mientras se siga necesitando corregir).
            if now - self.last_follow_time >= FOLLOW_TURN_BURST_COOLDOWN:
                angle = TURN_TOWARD_IMAGE_LEFT if desired_state == "left" else TURN_TOWARD_IMAGE_RIGHT
                command = ("drive_burst", angle, FOLLOW_TURN_BURST_DURATION)
                self.follow_state = desired_state
                self.last_follow_time = now

        elif (desired_state != self.follow_state
                and now - self.last_follow_time >= FOLLOW_MIN_COMMAND_INTERVAL):
            if desired_state == "forward":
                command = ("hold", 0, FOLLOW_FORWARD_POWER)
            elif desired_state == "backward":
                command = ("hold", 180, FOLLOW_BACKWARD_POWER)
            elif desired_state == "stop":
                command = ("stop", None)
            self.follow_state = desired_state
            self.last_follow_time = now

        return command


class R2Client:
    """Cliente WebSocket en un hilo propio; Tkinter nunca queda bloqueado."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.control_ws = None
        self.video_ws = None
        self.video_task = None
        self.control_reader_task = None
        self.keepalive_task = None
        self.wdt_task = None
        self.move_repeat_task = None
        self.head_repeat_task = None
        self.is_video_running = False
        self.last_frame_time = 0.0
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.processor = VisionProcessor()
        self.vision_mode = "Desactivado"
        self.seq = 10

    def start(self):
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def connect(self):
        self.submit(self._connect_control())

    async def _connect_control(self):
        if self.control_ws is not None:
            self.events.put(("status", "Control ya conectado"))
            return
        try:
            ws = await websockets.connect(CONTROL_URI, open_timeout=8, ping_interval=3, ping_timeout=5)
            self.control_ws = ws
            await self._send_raw({"cmd": "grantAccess", "uuid": CLIENT_UUID, "device_name": "PC-R2D2", "seq": 1})
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
                response = json.loads(raw)
                if response.get("cmd") == "grantAccess" and response.get("seq") == 1:
                    if response.get("resultCode") != 0:
                        raise RuntimeError(f"Autenticación rechazada: {response}")
                    break
            else:
                raise TimeoutError("El R2 no confirmó grantAccess")
            await self._send_raw({"cmd": "user_control", "enable": True})
            self.keepalive_task = asyncio.create_task(self._keepalive())
            self.wdt_task = asyncio.create_task(self._reset_wdt_loop())
            self._publish_robot(response.get("robot"))
            self.control_reader_task = asyncio.create_task(self._control_reader(ws))
            self.events.put(("status", "✓ Control conectado"))
        except Exception as exc:
            self.events.put(("error", f"Control: {exc}"))
            if self.control_ws:
                await self.control_ws.close()
            self.control_ws = None

    def _publish_robot(self, robot):
        if isinstance(robot, dict) and isinstance(robot.get("battery"), (int, float)):
            level = max(0, min(100, int(robot["battery"])))
            self.events.put(("battery", level))

    async def _control_reader(self, ws):
        """Consume los estados gin periódicos sin bloquear los controles."""
        try:
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if message.get("cmd") == "gin":
                    self._publish_robot(message.get("robot"))
        except Exception:
            pass
        finally:
            if self.control_ws is ws:
                self.control_ws = None
                self.events.put(("status", "Control desconectado"))

    async def _keepalive(self):
        try:
            while self.control_ws is not None:
                await asyncio.sleep(8)
                await self._send_raw({"cmd": "user_control", "enable": True})
        except Exception:
            pass

    async def _reset_wdt_loop(self):
        """La placa motora tiene su propio watchdog interno: si no se
        refresca periódicamente el comando "reset-wdt", asume que se
        perdió la comunicación y corta el movimiento como medida de
        seguridad, aunque el cliente siga mandando "move". Sin este
        refresco, mantener pulsado un botón del pad avanzaba un poco y
        luego se paraba solo a los pocos segundos."""
        try:
            while self.control_ws is not None:
                await self._send_raw({"cmd": "reset-wdt"})
                await asyncio.sleep(2)
        except Exception:
            pass

    async def _send_raw(self, message):
        if self.control_ws is None:
            raise RuntimeError("El control no está conectado")
        await self.control_ws.send(json.dumps(message, separators=(",", ":")) + "\n")

    async def _send_raw_safe(self, message):
        try:
            await self._send_raw(message)
        except Exception:
            pass

    def send(self, message):
        self.submit(self._send_logged(message))

    async def _send_logged(self, message):
        try:
            await self._send_raw(message)
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def burst_drive(self, angle, power=35, duration=0.25):
        async def action():
            await self._send_raw({"cmd": "move", "power": power, "angle": angle})
            await asyncio.sleep(duration)
            await self._send_raw({"cmd": "move", "power": 0, "angle": 0})
        self.submit(action())

    async def _start_repeat(self, task_attr, message, interval=0.3):
        """Reenvía `message` cada `interval` segundos hasta que se cancele.
        El comando "move" (y "head-dir") del robot tiene efecto de corta
        duración: la app oficial no manda un único comando y espera, sino
        que lo repite cada 300ms mientras el botón está pulsado. Sin este
        reenvío, el movimiento se corta solo al poco de empezar."""
        old = getattr(self, task_attr)
        if old and not old.done():
            old.cancel()

        async def loop():
            try:
                while True:
                    await self._send_raw_safe(message)
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        setattr(self, task_attr, asyncio.create_task(loop()))

    async def _stop_repeat(self, task_attr, stop_message=None):
        task = getattr(self, task_attr)
        was_running = task is not None and not task.done()
        if was_running:
            task.cancel()
            try:
                await task
            except Exception:
                pass
        setattr(self, task_attr, None)
        if was_running and stop_message:
            await self._send_raw_safe(stop_message)

    def start_drive(self, angle, power=35):
        """Movimiento continuo mientras se mantenga pulsado el botón del
        pad (a diferencia de burst_drive, no se para solo)."""
        self.submit(self._start_repeat("move_repeat_task", {"cmd": "move", "power": power, "angle": angle}))

    def stop_drive(self):
        self.submit(self._stop_repeat("move_repeat_task", {"cmd": "move", "power": 0, "angle": 0}))

    def start_head(self, direction):
        self.submit(self._start_repeat("head_repeat_task", {"cmd": "head-dir", "dir": direction}))

    def stop_head(self):
        self.submit(self._stop_repeat("head_repeat_task", {"cmd": "head-dir", "dir": 0}))

    def stop_all(self):
        self.send({"cmd": "move", "power": 0, "angle": 0})
        self.send({"cmd": "head-dir", "dir": 0})

    def start_video(self):
        self.submit(self._restart_video())

    async def _restart_video(self):
        """Cancela de verdad cualquier tarea de vídeo anterior (aunque esté
        colgada esperando una conexión muerta, p.ej. tras un corte de WiFi)
        antes de abrir una nueva. Sin esto, pulsar "Iniciar cámara" mientras
        la tarea anterior seguía viva (pero atascada) no hacía nada, o
        arrancaba una segunda conexión en paralelo que el robot rechazaba
        con 421 porque la primera seguía registrada como cliente activo."""
        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            try:
                await self.video_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._reset_follow()
        self.is_video_running = False
        self.video_task = asyncio.create_task(self._video_loop())

    def stop_video(self):
        self.submit(self._restart_video_stop())

    async def _restart_video_stop(self):
        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
            try:
                await self.video_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._reset_follow()
        self.is_video_running = False
        self.events.put(("camera_clear", None))
        self.events.put(("status", "Cámara detenida"))

    async def _video_loop(self):
        self.is_video_running = True
        watchdog_task = asyncio.create_task(self._follow_watchdog())
        try:
            # La app oficial siempre autentica y activa el control ANTES de abrir
            # el socket de vídeo (VideoControlActivity.connectVideoStream -> sendUserControl(true)).
            if self.control_ws is None:
                self.events.put(("status", "Autenticando control antes de abrir la cámara..."))
                await self._connect_control()
                if self.control_ws is None:
                    self.events.put(("error", "Cámara: no se pudo autenticar el control; no se abre la cámara"))
                    return

            max_attempts = 4
            retry_delay = 3
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    self.events.put(("status", f"Cámara ocupada, reintentando ({attempt}/{max_attempts})..."))
                    await asyncio.sleep(retry_delay)

                self.events.put(("status", "Conectando cámara..."))
                self.video_ws = await websockets.connect(VIDEO_URI, open_timeout=8, ping_interval=3, ping_timeout=5)

                first_frame_seen = False
                accepted_msg_seen = False
                got_421 = False
                async for raw_frame in self.video_ws:
                    if isinstance(raw_frame, bytes):
                        if not first_frame_seen:
                            first_frame_seen = True
                            self.events.put(("status", "✓ Cámara conectada"))
                        self.last_frame_time = time.time()

                        nparr = np.frombuffer(raw_frame, np.uint8)
                        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if cv_img is not None:
                            # Rotación corregida a 90° en sentido horario
                            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

                            if self.vision_mode != "Desactivado":
                                # La inferencia de MediaPipe es CPU-intensiva y
                                # bloqueante; se manda a un hilo aparte para no
                                # congelar este bucle (que también atiende los
                                # pings/pongs de los WebSockets) mientras dura.
                                cv_img, command = await asyncio.to_thread(
                                    self.processor.process_frame, cv_img, self.vision_mode
                                )
                                if command:
                                    self._execute_auto_command(command)

                            try:
                                self.frames.get_nowait()
                            except queue.Empty:
                                pass
                            self.frames.put_nowait(cv_img)
                    else:
                        # Mensajes de texto del robot: "enter video socket" al aceptar,
                        # o {"cmd":"streaming","resultCode":421} si ya hay otro cliente conectado.
                        print(f"[cámara] mensaje de texto del robot: {raw_frame!r}")
                        if raw_frame == "enter video socket":
                            accepted_msg_seen = True
                            continue
                        try:
                            message = json.loads(raw_frame)
                        except json.JSONDecodeError:
                            continue
                        if message.get("cmd") == "streaming" and message.get("resultCode") == 421:
                            got_421 = True

                if first_frame_seen:
                    return  # sesión de vídeo en marcha con normalidad

                close_code = getattr(self.video_ws, "close_code", None)
                close_reason = getattr(self.video_ws, "close_reason", None)
                print(f"[cámara] intento {attempt}/{max_attempts} sin vídeo. "
                      f"accepted_msg_seen={accepted_msg_seen} got_421={got_421} "
                      f"close_code={close_code} close_reason={close_reason!r}")

                if got_421 and attempt < max_attempts:
                    continue  # probablemente una conexión fantasma que aún no ha caducado; reintentar

                if got_421:
                    self.events.put((
                        "error",
                        "Cámara: el robot sigue diciendo que ya hay otro cliente conectado (resultCode 421) "
                        f"tras {max_attempts} intentos. Cierra la app móvil u otras instancias de este script; "
                        "si el corte fue por WiFi, puede que necesites esperar más o reiniciar el robot.",
                    ))
                elif accepted_msg_seen:
                    self.events.put((
                        "error",
                        f"Cámara: el robot aceptó la conexión ('enter video socket') pero nunca envió vídeo "
                        f"y luego cerró (code={close_code}, reason={close_reason!r}). "
                        "Puede que la cámara física del robot no esté arrancando en su lado.",
                    ))
                else:
                    self.events.put((
                        "error",
                        f"Cámara: el robot cerró la conexión sin aceptarla ni enviar vídeo "
                        f"(code={close_code}, reason={close_reason!r}). "
                        "Revisa la consola para ver el mensaje de texto exacto, si lo hubo.",
                    ))
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.events.put(("error", f"Cámara: {exc}"))
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except Exception:
                pass
            if self.video_ws is not None:
                try:
                    await self.video_ws.close()
                except Exception:
                    pass
            self.is_video_running = False
            self.video_ws = None
            # Si "Sígueme" se quedó a media maniobra al cortarse el vídeo,
            # que no se quede el robot moviéndose sin control.
            await self._reset_follow()

    async def _reset_follow(self):
        """Para el cuerpo de "Sígueme" si estaba en movimiento, dejando el
        estado de seguimiento limpio."""
        if self.processor.follow_state is not None:
            self.processor.follow_state = None
            await self._stop_repeat("move_repeat_task", {"cmd": "move", "power": 0, "angle": 0})

    async def _follow_watchdog(self):
        """Parada de emergencia si "Sígueme" está en medio de un movimiento
        continuo y dejan de llegar fotogramas (WiFi cortado, robot colgado,
        etc.). Sin esto, un "move" continuo sin llegar más frames que lo
        corrijan podría dejar al robot avanzando indefinidamente."""
        try:
            while True:
                await asyncio.sleep(FOLLOW_WATCHDOG_INTERVAL)
                stale = time.time() - self.last_frame_time > FOLLOW_WATCHDOG_STALE_FRAMES
                if (self.vision_mode == "Sígueme"
                        and self.processor.follow_state not in (None, "stop")
                        and stale):
                    await self._reset_follow()
                    self.events.put(("status", "Sígueme: sin vídeo reciente, parada de seguridad"))
        except asyncio.CancelledError:
            pass

    def _execute_auto_command(self, command):
        kind = command[0]
        if kind == "hold":
            # Movimiento continuo: se repite cada 300ms mientras dure este
            # estado (a diferencia de burst_drive, que se para solo). El
            # watchdog de _video_loop lo corta si dejan de llegar fotogramas.
            _, angle, power = command
            self.submit(self._start_repeat("move_repeat_task", {"cmd": "move", "power": power, "angle": angle}))
            return
        # Cualquier otro comando reemplaza un "hold" de avance/retroceso en curso.
        self.submit(self._stop_repeat("move_repeat_task", {"cmd": "move", "power": 0, "angle": 0}))
        if kind == "stop":
            self.send({"cmd": "move", "power": 0, "angle": 0})
        elif kind == "mode":
            self.send({"cmd": "mode", "mode": command[1]})
        elif kind == "sound":
            self.send({"cmd": "play_sound", "sound_id": command[1], "interrupt": 1})
        elif kind == "drive_burst":
            _, angle, duration = command
            self.burst_drive(angle=angle, duration=duration)

    def close(self):
        """Desconecta limpiamente del robot. Devuelve el Future de la tarea de
        cierre para que el llamador pueda esperarlo antes de matar el proceso
        (el hilo del bucle asyncio es daemon y se mata en seco al salir, así
        que si no se espera aquí, el robot puede quedarse sin el cierre
        limpio del WebSocket y depender de su propio timeout de 5s, o incluso
        no liberar la cámara si el proceso muere antes)."""
        async def close_all():
            if self.move_repeat_task and not self.move_repeat_task.done():
                self.move_repeat_task.cancel()
                try:
                    await self.move_repeat_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self.head_repeat_task and not self.head_repeat_task.done():
                self.head_repeat_task.cancel()
                try:
                    await self.head_repeat_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self.video_task and not self.video_task.done():
                self.video_task.cancel()
                try:
                    await self.video_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self.control_ws:
                try:
                    await self._send_raw({"cmd": "move", "power": 0, "angle": 0})
                    await self._send_raw({"cmd": "head-dir", "dir": 0})
                    await self._send_raw({"cmd": "user_control", "enable": False})
                    await self.control_ws.close()
                except Exception:
                    pass
            if self.video_ws:
                try:
                    await self.video_ws.close()
                except Exception:
                    pass
        return self.submit(close_all())


class R2Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("R2-D2 Control Center + Visión AI")
        self.geometry("1150x780")
        self.minsize(980, 620)
        self.client = R2Client()
        self.client.start()
        self.status = tk.StringVar(value="Sin conectar")
        self.battery_text = tk.StringVar(value="Batería: —")
        self.battery_value = tk.IntVar(value=0)
        self.sound = tk.StringVar(value="0")
        self.vision_mode_var = tk.StringVar(value="Desactivado")
        self.drive_power = tk.IntVar(value=35)
        self.drive_power_text = tk.StringVar(value="35")
        self._image = None
        self._build()
        self._bind_keyboard()
        self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    @staticmethod
    def _hold_button(parent, text, on_press, on_release, **kwargs):
        """Botón tipo "pulsa y mantén": arranca el movimiento en
        ButtonPress y lo para en ButtonRelease, en vez de una ráfaga de
        duración fija al hacer clic."""
        button = ttk.Button(parent, text=text, **kwargs)
        button.bind("<ButtonPress-1>", lambda event: on_press())
        button.bind("<ButtonRelease-1>", lambda event: on_release())
        return button

    def _bind_key_hold(self, keysym, on_press, on_release):
        """Como _hold_button pero para una tecla: arranca en KeyPress y para
        en KeyRelease. El sistema operativo genera pares release+press muy
        seguidos mientras mantienes la tecla (auto-repetición), que si no se
        filtran se verían como parpadeos de para/arranca en vez de un
        movimiento continuo. Por eso el "release" real se retrasa 50ms y se
        cancela si llega un nuevo "press" de esa tecla antes de que se
        cumpla (señal de que era solo auto-repetición, no una soltada real).
        bind_all para que funcione tenga el foco el widget que tenga (así no
        hace falta hacer clic en el pad para que las flechas funcionen)."""
        state = {"pressed": False, "release_id": None}

        def handle_press(event):
            if state["release_id"] is not None:
                self.after_cancel(state["release_id"])
                state["release_id"] = None
            if not state["pressed"]:
                state["pressed"] = True
                on_press()

        def handle_release(event):
            def real_release():
                state["pressed"] = False
                state["release_id"] = None
                on_release()
            state["release_id"] = self.after(50, real_release)

        self.bind_all(f"<KeyPress-{keysym}>", handle_press)
        self.bind_all(f"<KeyRelease-{keysym}>", handle_release)

    def _bind_keyboard(self):
        self._bind_key_hold("Up", lambda: self.client.start_drive(0, power=self.drive_power.get()), self.client.stop_drive)
        self._bind_key_hold("Down", lambda: self.client.start_drive(180, power=self.drive_power.get()), self.client.stop_drive)
        self._bind_key_hold("Left", lambda: self.client.start_drive(-90, power=self.drive_power.get()), self.client.stop_drive)
        self._bind_key_hold("Right", lambda: self.client.start_drive(90, power=self.drive_power.get()), self.client.stop_drive)
        for key in ("z", "Z"):
            self._bind_key_hold(key, lambda: self.client.start_head(-1), self.client.stop_head)
        for key in ("x", "X"):
            self._bind_key_hold(key, lambda: self.client.start_head(1), self.client.stop_head)

    def _build(self):
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Conectar control", command=self.client.connect).pack(side="left")
        ttk.Button(toolbar, text="Iniciar cámara", command=self.client.start_video).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Detener cámara", command=self.client.stop_video).pack(side="left")

        ttk.Label(toolbar, text="Modo Visión:").pack(side="left", padx=(15, 5))
        vision_combo = ttk.Combobox(
            toolbar,
            textvariable=self.vision_mode_var,
            values=["Desactivado", "Gestos con la mano", "Sígueme"],
            state="readonly",
            width=20,
        )
        vision_combo.pack(side="left")
        vision_combo.bind("<<ComboboxSelected>>", self._on_vision_change)

        ttk.Label(toolbar, textvariable=self.status).pack(side="right")
        ttk.Progressbar(toolbar, variable=self.battery_value, maximum=100, length=120).pack(side="right", padx=(12, 4))
        ttk.Label(toolbar, textvariable=self.battery_text).pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        controls = ttk.Frame(main, padding=8)
        camera = ttk.Frame(main, padding=8)
        main.add(controls, weight=1)
        main.add(camera, weight=2)

        ttk.Label(controls, text="Movimiento", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        speed_row = ttk.Frame(controls)
        speed_row.pack(fill="x", pady=(4, 2))
        ttk.Label(speed_row, text="Velocidad:").pack(side="left")
        ttk.Scale(
            speed_row, from_=10, to=100, orient="horizontal",
            variable=self.drive_power, command=self._on_power_change,
        ).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(speed_row, textvariable=self.drive_power_text, width=4).pack(side="left")

        pad = ttk.Frame(controls)
        pad.pack(pady=5)
        self._hold_button(pad, "▲", lambda: self.client.start_drive(0, power=self.drive_power.get()), self.client.stop_drive, width=8).grid(row=0, column=1, padx=2, pady=2)
        self._hold_button(pad, "◀", lambda: self.client.start_drive(-90, power=self.drive_power.get()), self.client.stop_drive, width=8).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(pad, text="■ STOP", width=8, command=self.client.stop_all).grid(row=1, column=1, padx=2, pady=2)
        self._hold_button(pad, "▶", lambda: self.client.start_drive(90, power=self.drive_power.get()), self.client.stop_drive, width=8).grid(row=1, column=2, padx=2, pady=2)
        self._hold_button(pad, "▼", lambda: self.client.start_drive(180, power=self.drive_power.get()), self.client.stop_drive, width=8).grid(row=2, column=1, padx=2, pady=2)

        ttk.Label(controls, text="Cabeza", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(12, 0))
        head = ttk.Frame(controls)
        head.pack(pady=5)
        self._hold_button(head, "◀ Cabeza", lambda: self.client.start_head(-1), self.client.stop_head).pack(side="left", padx=3)
        self._hold_button(head, "Cabeza ▶", lambda: self.client.start_head(1), self.client.stop_head).pack(side="left", padx=3)

        ttk.Label(controls, text="Funciones", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(12, 0))
        funcs = ttk.Frame(controls)
        funcs.pack(fill="x", pady=4)
        actions = [("Brazo", 16), ("Sable", 6), ("Proyector 1", 19), ("Proyector 2", 20),
                   ("LCD corta", 17), ("LCD larga", 18), ("Patrulla", 9), ("Baile", 10)]
        for index, (label, mode) in enumerate(actions):
            ttk.Button(funcs, text=label, command=lambda m=mode: self.client.send({"cmd": "mode", "mode": m})).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2
            )
        ttk.Button(controls, text="Restaurar luces normales", command=lambda: self.client.send(
            {"cmd": "led", "r": 2, "b": 2, "y": 2, "g": 1}
        )).pack(fill="x", pady=(6, 2))
        audio = ttk.Frame(controls)
        audio.pack(fill="x", pady=2)
        ttk.Button(audio, text="Silenciar", command=lambda: self.client.send({"cmd": "mute", "enable": True, "seq": 30})).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(audio, text="Activar audio", command=lambda: self.client.send({"cmd": "mute", "enable": False, "seq": 31})).pack(side="left", expand=True, fill="x", padx=(2, 0))

        ttk.Label(controls, text="Sonidos", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(12, 0))
        sound_row = ttk.Frame(controls)
        sound_row.pack(fill="x", pady=4)
        ttk.Combobox(sound_row, textvariable=self.sound, values=[str(i) for i in range(19)], width=6, state="readonly").pack(side="left")
        ttk.Button(sound_row, text="Reproducir", command=self._play_sound).pack(side="left", padx=5)

        ttk.Label(controls, text="Gestos con la mano", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(15, 0))
        ttk.Label(
            controls,
            justify="left",
            text=(
                "Puño cerrado → Parar\n"
                "Palma abierta → Saludo\n"
                "Pulgar arriba → Baile\n"
                "Seña de victoria → Sable"
            ),
        ).pack(anchor="w")

        ttk.Label(controls, text="Sígueme", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(15, 0))
        ttk.Label(
            controls,
            wraplength=260,
            justify="left",
            text="El robot detecta tu silueta (funciona también de espaldas) y gira para mantenerte centrado, avanzando o retrocediendo para quedarse a media distancia.",
        ).pack(anchor="w")

        ttk.Label(camera, text="Cámara del R2-D2 — WebSocket :12121", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.video = ttk.Label(camera, text="Pulsa “Iniciar cámara” para recibir el vídeo", anchor="center")
        self.video.pack(fill="both", expand=True, pady=8)

    def _play_sound(self):
        self.client.send({"cmd": "play_sound", "sound_id": int(self.sound.get()), "interrupt": 1})

    def _on_power_change(self, value):
        power = round(float(value))
        self.drive_power.set(power)
        self.drive_power_text.set(str(power))

    def _on_vision_change(self, event):
        new_mode = self.vision_mode_var.get()
        if self.client.vision_mode == "Sígueme" and new_mode != "Sígueme":
            # Si estaba en medio de un movimiento continuo, no dejarlo así al
            # salir del modo. _reset_follow vive en el hilo del bucle
            # asyncio, así que se despacha con submit() en vez de tocar su
            # estado directamente desde el hilo de Tkinter.
            self.client.submit(self.client._reset_follow())
        self.client.vision_mode = new_mode

    def _poll(self):
        try:
            while True:
                kind, message = self.client.events.get_nowait()
                if kind == "battery":
                    self.battery_value.set(message)
                    self.battery_text.set(f"Batería: {message}%")
                elif kind == "camera_clear":
                    self._image = None
                    self.video.configure(image="", text="Cámara detenida")
                else:
                    self.status.set(message)
        except queue.Empty:
            pass
        try:
            cv_img = self.client.frames.get_nowait()
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            img_pil.thumbnail((720, 600))
            self._image = ImageTk.PhotoImage(img_pil)
            self.video.configure(image=self._image, text="")
        except queue.Empty:
            pass
        except Exception as exc:
            self.status.set(f"Imagen: {exc}")
        self.after(40, self._poll)

    def _close(self):
        future = self.client.close()
        try:
            future.result(timeout=3)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    R2Gui().mainloop()
