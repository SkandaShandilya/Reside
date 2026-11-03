"""
Video Stream Service â€” Phase 2

Manages concurrent video stream connections for all configured cameras.

Responsibilities:
- Open/maintain OpenCV VideoCapture for RTSP, webcam, and video file sources
- Provide async frame generators for MJPEG streaming
- Health monitoring (connection drops, FPS tracking)
- Thread-safe frame buffer per camera

Architecture:
  Camera source (RTSP / webcam / file)
        â†“
  VideoCapture thread (runs in executor)
        â†“
  FrameBuffer (latest N frames)
        â†“
  MJPEG generator (streams to browser via HTTP)
"""
import os
# Configure low-latency FFMPEG settings for RTSP/network captures
# MUST be set before cv2 is imported to take effect
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0"

import asyncio
import cv2
import queue as _queue
import sys
import time
import threading
import logging
from typing import Optional, Dict, AsyncGenerator
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from ai_service.config import AIConfig

logger = logging.getLogger(__name__)

# Activity label ai_service.inference_engine's fire_smoke_state pipeline
# emits (see AlertEvent.activity / fire_smoke_event_updates' "activity"
# field) — used to route this specifically to the fire/smoke-overlay
# buffer and the pre+active+post incident recorder below, instead of the
# behaviour buffer / single-snapshot clip everything else gets. Fire and
# smoke are combined into this one activity now (see
# inference_engine.py's combined-event construction) — kept as a tuple
# (not a bare string) since this is iterated elsewhere in this file.
FIRE_SMOKE_LABELS = ("Fire/Smoke Detected",)
# Hard cap on frames held per open incident (a deque, so once full the
# OLDEST frames are dropped first — guaranteeing the POST-event tail is
# always captured even for an incident that runs far longer than this).
# At typical native display fps (15-30), this covers roughly 1-2 minutes
# of the MOST RECENT activity, which is what a reviewer actually needs to
# see; kept as a cap (not unbounded) so a very long-running fire can't grow
# memory usage without bound.
MAX_FIRE_SMOKE_INCIDENT_FRAMES = 1800


@dataclass
class FrameBuffer:
    """Thread-safe circular frame buffer for a single camera."""
    camera_id: int
    maxlen: int = 30
    _frames: deque = field(default_factory=lambda: deque(maxlen=30))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_frame_time: float = field(default_factory=time.time)
    _fps: float = 0.0
    _frame_count: int = 0
    _is_alive: bool = False

    def push(self, frame: bytes) -> None:
        now = time.time()
        with self._lock:
            self._frames.append(frame)
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                self._fps = 1.0 / elapsed
            self._last_frame_time = now
            self._frame_count += 1
            self._is_alive = True

    def latest(self) -> Optional[bytes]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def recent(self) -> list:
        """
        Snapshot of every JPEG frame currently held (already-rendered,
        overlay included) — used to build a short "leading up to the
        event" clip when an alert fires, without needing a second buffer.
        """
        with self._lock:
            return list(self._frames)

    @property
    def fps(self) -> float:
        with self._lock:
            return round(self._fps, 1)

    @property
    def is_alive(self) -> bool:
        with self._lock:
            # 15s grace window â€” YOLO model load can take 8-12s on first run
            return self._is_alive and (time.time() - self._last_frame_time) < 15.0


@dataclass
class StreamInfo:
    """Metadata about a running camera stream."""
    camera_id: int
    source_url: str
    source_type: str  # rtsp / webcam / file / network
    started_at: datetime
    buffer: FrameBuffer
    # Second, independently-rendered stream showing ONLY the fire/smoke
    # overlay (no person boxes/behaviour HUD) — for the dedicated Fire &
    # Smoke tab. Same raw frames, separate render pass (see
    # inference_engine.render_fire_smoke_only()), so the two tabs never
    # interfere with each other's overlay content.
    fire_smoke_buffer: Optional[FrameBuffer] = None
    thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    error: Optional[str] = None
    consecutive_failures: int = 0
    # Fire/Smoke incident clip recording (pre+active+post — see
    # _update_fire_smoke_incidents). Keyed by activity label ("Fire
    # Detected"/"Smoke Detected"). `signals` is written by the AI worker
    # thread and consumed/cleared by the capture thread; `incidents` is
    # only ever touched by the capture thread, so it needs no lock.
    fire_smoke_incident_signals: Dict[str, str] = field(default_factory=dict)
    fire_smoke_incident_signals_lock: threading.Lock = field(default_factory=threading.Lock)
    fire_smoke_incidents: Dict[str, dict] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        return (
            self.thread is not None
            and self.thread.is_alive()
            and self.buffer.is_alive
        )


class VideoStreamService:
    """
    Central manager for all active camera video streams.

    Usage:
        service = VideoStreamService()
        await service.start_stream(camera_id=1, source_url="rtsp://...", source_type="rtsp")
        async for jpeg_frame in service.frame_generator(camera_id=1):
            yield jpeg_frame
    """

    def __init__(self, ai_fps: int = 60, jpeg_quality: int = 88):
        self.ai_fps = ai_fps
        self.jpeg_quality = jpeg_quality
        # Inference runs every N capture frames (2 = every other frame for max speed)
        self.ai_inference_every_n_frames: int = 2
        self._streams: Dict[int, StreamInfo] = {}
        self._lock = threading.Lock()
        # Per-camera "AI Detection Processing" toggle, checked live by the
        # capture loop every frame so flipping it in Camera Settings takes
        # effect immediately without restarting the stream.
        self._ai_enabled: Dict[int, bool] = {}
        self._ai_enabled_lock = threading.Lock()
        # Per-camera restricted/monitoring zones, checked live by the AI
        # worker so creating/deleting a zone takes effect immediately
        # without restarting the stream.
        self._zones: Dict[int, list] = {}
        self._zones_lock = threading.Lock()
        # Per-camera Fire/Smoke detection toggles — INDEPENDENT of
        # `_ai_enabled` above (which gates the whole behaviour pipeline).
        # Read from Camera.detection_config's "fire_enabled"/"smoke_enabled"
        # keys (default True), checked live so a Camera Settings change
        # applies without restarting the stream.
        self._fire_smoke_enabled: Dict[int, tuple] = {}
        self._fire_smoke_enabled_lock = threading.Lock()
        # Latest FrameAnalysis.telemetry dict per camera (activity/fire/
        # smoke state + confidences) — polled by the frontend for
        # non-video status displays (e.g. a Fire & Smoke tab) without
        # needing to decode the MJPEG stream itself.
        self._telemetry: Dict[int, dict] = {}
        self._telemetry_lock = threading.Lock()
        # System-wide AI analysis rate (see AIConfig.AI_ANALYSIS_FPS docs).
        # A plain attribute, not per-camera — read fresh every cycle by each
        # camera's AI worker so a System Settings change applies live.
        self.ai_analysis_fps: int = AIConfig.AI_ANALYSIS_FPS

    def set_ai_analysis_fps(self, fps: int) -> None:
        """Change how often (per second) each camera's AI worker runs a detection pass."""
        self.ai_analysis_fps = max(1, fps)

    def set_ai_enabled(self, camera_id: int, enabled: bool) -> None:
        """Turn AI detection/overlay on or off for a camera's live stream."""
        with self._ai_enabled_lock:
            self._ai_enabled[camera_id] = enabled

    def is_ai_enabled(self, camera_id: int) -> bool:
        with self._ai_enabled_lock:
            return self._ai_enabled.get(camera_id, True)

    def set_zones(self, camera_id: int, zones: list) -> None:
        """Update the zones a running/future stream for this camera uses."""
        with self._zones_lock:
            self._zones[camera_id] = zones

    def get_zones(self, camera_id: int) -> list:
        with self._zones_lock:
            return self._zones.get(camera_id, [])

    def set_fire_smoke_enabled(self, camera_id: int, fire_enabled: bool, smoke_enabled: bool) -> None:
        """Independently enable/disable fire and smoke detection for a
        camera's live stream — does not affect the behaviour pipeline."""
        with self._fire_smoke_enabled_lock:
            self._fire_smoke_enabled[camera_id] = (fire_enabled, smoke_enabled)

    def get_fire_smoke_enabled(self, camera_id: int) -> tuple:
        with self._fire_smoke_enabled_lock:
            return self._fire_smoke_enabled.get(camera_id, (True, True))

    def get_telemetry(self, camera_id: int) -> dict:
        with self._telemetry_lock:
            return dict(self._telemetry.get(camera_id, {}))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_stream(
        self,
        camera_id: int,
        source_url: str,
        source_type: str = "rtsp",
        username: str = "",
        password: str = "",
        ai_enabled: bool = True,
        zones: Optional[list] = None,
        detection_config: Optional[dict] = None,
    ) -> bool:
        """
        Start a background capture thread for a camera.
        Returns True if started successfully, False if already running.
        """
        self.set_ai_enabled(camera_id, ai_enabled)
        if zones is not None:
            self.set_zones(camera_id, zones)
        # Fire/Smoke toggles are independent of `ai_enabled` (which gates
        # the behaviour pipeline) — read from Camera.detection_config,
        # defaulting both to True when unset so existing cameras keep
        # working without needing a config update.
        dc = detection_config or {}
        self.set_fire_smoke_enabled(
            camera_id,
            fire_enabled=dc.get("fire_enabled", True),
            smoke_enabled=dc.get("smoke_enabled", True),
        )
        with self._lock:
            existing = self._streams.get(camera_id)
            # A capture thread that is alive but hasn't produced its first
            # frame yet (slow webcam open, reconnecting) is NOT "not
            # running" — treating it that way let every WebSocket
            # (re)connect spawn another thread for the same camera, and
            # they then fought over the one physical device ("Cannot open").
            if existing is not None and (
                existing.is_running
                or (existing.thread is not None and existing.thread.is_alive()
                    and existing.stop_event is not None and not existing.stop_event.is_set())
            ):
                logger.info(f"Camera {camera_id} stream already running.")
                return False

        # Build the OpenCV-compatible URL
        capture_url = self._build_capture_url(source_url, source_type, username, password)
        stop_event = threading.Event()
        buffer = FrameBuffer(camera_id=camera_id, maxlen=self.ai_fps * 3)
        fire_smoke_buffer = FrameBuffer(camera_id=camera_id, maxlen=self.ai_fps * 3)

        info = StreamInfo(
            camera_id=camera_id,
            source_url=capture_url,
            source_type=source_type,
            started_at=datetime.utcnow(),
            buffer=buffer,
            fire_smoke_buffer=fire_smoke_buffer,
            stop_event=stop_event,
        )

        thread = threading.Thread(
            target=self._capture_loop,
            args=(info,),
            daemon=True,
            name=f"cam-{camera_id}",
        )
        info.thread = thread

        with self._lock:
            self._streams[camera_id] = info

        thread.start()
        logger.info(f"Camera {camera_id} capture thread started ({source_type}).")
        return True

    def stop_stream(self, camera_id: int) -> None:
        """Signal a camera stream to stop."""
        with self._lock:
            info = self._streams.get(camera_id)
        if info and info.stop_event:
            info.stop_event.set()
            logger.info(f"Camera {camera_id} stream stop requested.")

    def stop_all(self) -> None:
        """Stop all active streams."""
        with self._lock:
            camera_ids = list(self._streams.keys())
        for cid in camera_ids:
            self.stop_stream(cid)

    def get_status(self, camera_id: int) -> dict:
        """Return current stream status info for a camera."""
        with self._lock:
            info = self._streams.get(camera_id)
        if not info:
            return {"camera_id": camera_id, "running": False, "fps": 0, "error": None}
        return {
            "camera_id": camera_id,
            "running": info.is_running,
            "fps": info.buffer.fps,
            "source_type": info.source_type,
            "started_at": info.started_at.isoformat(),
            "error": info.error,
            "consecutive_failures": info.consecutive_failures,
        }

    def get_all_status(self) -> list:
        """Return status for all tracked streams."""
        with self._lock:
            camera_ids = list(self._streams.keys())
        return [self.get_status(cid) for cid in camera_ids]

    async def frame_generator(
        self,
        camera_id: int,
        fps: int = 30,
        variant: str = "full",
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator yielding JPEG frames as multipart/x-mixed-replace.
        Immediately streams placeholder frames while the capture thread warms up,
        then switches to live frames as soon as the first one arrives.

        `variant`: "full" (default — every overlay) or "fire_smoke" (only
        the fire/smoke overlay, see StreamInfo.fire_smoke_buffer / the
        dedicated Fire & Smoke tab).
        """
        interval = 1.0 / fps
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        # Wait up to 30s for first real frame (YOLO load + file open)
        max_wait_seconds = 30
        waited = 0.0

        while True:
            with self._lock:
                info = self._streams.get(camera_id)

            frame = None
            if info is not None:
                source_buffer = info.fire_smoke_buffer if (variant == "fire_smoke" and info.fire_smoke_buffer) else info.buffer
                frame = source_buffer.latest()

            if frame is None:
                # Stream not ready yet â€” send placeholder and wait
                waited += interval
                if waited > max_wait_seconds:
                    logger.warning(f"Camera {camera_id}: no frame after {max_wait_seconds}s, stopping generator.")
                    break
                frame = self._placeholder_frame(camera_id)
            else:
                # Reset wait counter once frames are flowing
                waited = 0.0

            yield boundary + frame + b"\r\n"
            await asyncio.sleep(interval)

            # Stop only if explicitly stopped (not just because buffer went quiet briefly)
            if info and info.stop_event and info.stop_event.is_set():
                if info.thread and not info.thread.is_alive():
                    break

    async def test_connection(
        self,
        source_url: str,
        source_type: str = "rtsp",
        username: str = "",
        password: str = "",
        timeout: float = 8.0,
    ) -> dict:
        """
        Attempt to open a capture and grab one frame.
        Returns success/failure with details.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._blocking_test_connection,
            source_url, source_type, username, password, timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_loop(self, info: StreamInfo) -> None:
        """
        Background thread: reads frames at native source FPS, draws the
        latest available AI analysis onto EVERY frame, and pushes exactly
        one JPEG per frame into the FrameBuffer.

        AI analysis (YOLO/tracking/behaviour) runs in a SEPARATE worker
        thread since it can't keep up with native frame rate, but only
        that worker's *results* (bounding boxes, labels) are shared --
        never a fully-rendered frame. If both the capture loop and the AI
        worker pushed finished frames into the same buffer, the two would
        race and the stream would flicker between a fresh frame and a
        stale (older) annotated one, which reads as choppy/stuttering
        playback. Rendering overlays every frame in this single loop keeps
        video smooth even though the underlying analysis lags behind it.
        """
        from ai_service.inference_engine import inference_engine, FrameAnalysis

        backoff = 1.0
        max_backoff = 30.0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        # per-stream inference queue (depth 1 = lowest latency, always freshest frame)
        infer_q: "_queue.Queue" = _queue.Queue(maxsize=1)
        analysis_lock = threading.Lock()
        latest_analysis = FrameAnalysis()

        def _inference_worker():
            """Dedicated AI thread: pops freshest raw frame, updates the shared analysis."""
            nonlocal latest_analysis
            while True:
                cycle_start = time.time()
                item = infer_q.get()
                if item is None:          # sentinel -> exit
                    infer_q.task_done()
                    break
                raw_frame = item
                try:
                    zones = self.get_zones(info.camera_id)
                    fire_enabled, smoke_enabled = self.get_fire_smoke_enabled(info.camera_id)
                    result = inference_engine.analyze(
                        camera_id=info.camera_id, frame=raw_frame, zones=zones,
                        fire_enabled=fire_enabled, smoke_enabled=smoke_enabled,
                    )
                    with analysis_lock:
                        latest_analysis = result
                    with self._telemetry_lock:
                        self._telemetry[info.camera_id] = result.telemetry
                    if result.alert is not None:
                        # Persist + broadcast — see alert_dispatcher.py. This
                        # is the one place a detection actually becomes a
                        # SecurityEvent/Alert row and reaches the frontend;
                        # analyze() only ever computed the AlertEvent, it
                        # never had anywhere to send it before this call
                        # existed.
                        from app.services import alert_dispatcher
                        all_new_alerts = [result.alert] + list(result.additional_alerts)
                        for a in all_new_alerts:
                            # Fire/Smoke alerts use the fire/smoke-overlay
                            # buffer (shows the actual detection box) rather
                            # than the behaviour buffer, which is
                            # structurally incapable of drawing fire/smoke
                            # overlays at all (see render_behaviour_only's
                            # docstring) — using it here was the root cause
                            # of Fire/Smoke alert clips showing an unrelated
                            # behaviour overlay with no fire/smoke visible.
                            is_fire_smoke = a.activity in FIRE_SMOKE_LABELS
                            source_buffer = info.fire_smoke_buffer if (is_fire_smoke and info.fire_smoke_buffer) else info.buffer
                            alert_dispatcher.dispatch_alert_sync(a, source_buffer.recent())
                            if is_fire_smoke:
                                with info.fire_smoke_incident_signals_lock:
                                    info.fire_smoke_incident_signals[a.activity] = "start"
                    if result.behaviour_event_updates or result.fire_smoke_event_updates:
                        # Refresh/finalize an already-open event's row (Phase
                        # 2 edge-triggered lifecycle) — see
                        # ai_service/activity_state.py / fire_smoke_state.py
                        # + inference_engine.py. Never a new row/broadcast,
                        # just an UPDATE of peak/average/duration (and
                        # end_time on finalize). Behaviour and fire/smoke
                        # updates are dispatched through the same generic
                        # function but kept in separate lists all the way
                        # down, per the architectural mandate that the two
                        # subsystems stay independent.
                        from app.services import alert_dispatcher
                        for update in result.behaviour_event_updates:
                            alert_dispatcher.dispatch_event_update_sync(update)
                        for update in result.fire_smoke_event_updates:
                            alert_dispatcher.dispatch_event_update_sync(update)
                            # Signal the capture loop's incident recorder:
                            # "finalize" starts the post-event grace window
                            # (don't stop collecting frames yet), "reopen"
                            # cancels a pending close (the incident is
                            # still/again active).
                            if update["action"] in ("finalize", "reopen"):
                                with info.fire_smoke_incident_signals_lock:
                                    info.fire_smoke_incident_signals[update["activity"]] = (
                                        "end" if update["action"] == "finalize" else "start"
                                    )
                except Exception as ai_err:
                    logger.warning(f"AI inference error cam {info.camera_id}: {ai_err}")
                finally:
                    infer_q.task_done()

                # Throttle to the current AI_ANALYSIS_FPS setting — read fresh
                # each cycle so changing it in System Settings takes effect
                # immediately on already-running streams. Since infer_q
                # (maxsize=1) always holds the newest frame, whatever we pick
                # up next after this sleep is fresh, not a growing backlog.
                fps = self.ai_analysis_fps
                min_interval = 1.0 / fps if fps > 0 else 0.0
                remaining = min_interval - (time.time() - cycle_start)
                if remaining > 0:
                    time.sleep(remaining)

        worker = threading.Thread(target=_inference_worker, daemon=True,
                                  name=f"ai-{info.camera_id}")
        worker.start()

        while not info.stop_event.is_set():
            cap = None
            reader_thread: Optional[threading.Thread] = None
            reader_stop = threading.Event()
            frame_q: "_queue.Queue" = _queue.Queue(maxsize=1)
            try:
                # FFMPEG capture options are now set at the top of the file
                is_webcam = str(info.source_type).lower() == "webcam"

                if is_webcam and sys.platform.startswith("win"):
                    # OpenCV's default backend on Windows (MSMF) is well known
                    # to be slow to open and to occasionally stall mid-stream
                    # on USB/onboard webcams. DirectShow is far more reliable
                    # for this exact case.
                    cap = cv2.VideoCapture(info.source_url, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(info.source_url)
                if not cap.isOpened():
                    raise ConnectionError(f"Cannot open: {info.source_url}")

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if is_webcam:
                    # Ask the driver to deliver an already-modest frame size
                    # instead of capturing native resolution (often 1080p+ on
                    # modern webcams) and downscaling every single frame in
                    # software — cuts both USB bandwidth and per-frame CPU
                    # cost, the main cause of a webcam feed reading as
                    # slow/laggy rather than genuinely stuck.
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                info.error = None
                info.consecutive_failures = 0
                backoff = 1.0

                source_fps = cap.get(cv2.CAP_PROP_FPS)
                if source_fps <= 0 or source_fps > 120:
                    source_fps = 30.0
                frame_delay = 1.0 / source_fps

                # A webcam's source_url is a bare device index (e.g. "0"),
                # which — same as a local file path — never starts with
                # rtsp://\http://\https://, so it must be excluded from this
                # heuristic explicitly. Left unguarded, every webcam stream
                # was silently misclassified as a video file and run through
                # the file-replay branch below (including its loop-restart-
                # on-failure logic, and skipping the live-source watchdog
                # entirely) instead of proper live-stream handling.
                is_video_file = (
                    str(info.source_type).lower() != "webcam"
                    and (
                        str(info.source_type).lower() in ("upload", "file")
                        or str(info.source_url).lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'))
                        or not str(info.source_url).startswith(('rtsp://', 'http://', 'https://'))
                    )
                )
                logger.info(
                    f"Camera {info.camera_id}: capture opened "
                    f"({'file' if is_video_file else 'live'}, {source_fps:.1f} fps)."
                )

                # For live sources (webcam/RTSP/network/NVR — never a video
                # file, which never hangs the way a flaky camera driver can),
                # reads happen on a dedicated thread so a stalled cap.read()
                # can never freeze this loop forever: the main loop below
                # waits on frame_q with a timeout and treats a stall as a
                # connection failure, which the existing backoff-and-reopen
                # logic already recovers from. Previously a hung read here
                # just silently froze the stream with no way to recover —
                # this is what showed up as the live feed "getting stuck."
                if not is_video_file:
                    def _threaded_reader(_cap=cap, _q=frame_q, _stop=reader_stop):
                        while not _stop.is_set():
                            ok, frm = _cap.read()
                            if not ok:
                                try:
                                    _q.get_nowait()
                                except _queue.Empty:
                                    pass
                                try:
                                    _q.put_nowait((False, None))
                                except _queue.Full:
                                    pass
                                return
                            try:
                                _q.get_nowait()  # drop any stale unread frame
                            except _queue.Empty:
                                pass
                            try:
                                _q.put_nowait((True, frm))
                            except _queue.Full:
                                pass

                    reader_thread = threading.Thread(
                        target=_threaded_reader, daemon=True,
                        name=f"cam-{info.camera_id}-reader",
                    )
                    reader_thread.start()

                while not info.stop_event.is_set():
                    t0 = time.time()

                    if is_video_file:
                        ret, frame = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                cap.release()
                                cap = cv2.VideoCapture(info.source_url)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                ret, frame = cap.read()
                            # An uploaded video looping back to frame 0 is a
                            # hard scene cut (last frame -> first frame) that
                            # a real live camera never produces. Left alone,
                            # the fire/smoke detector's motion-based anti-
                            # false-positive gate compares that seam as if it
                            # were two temporally-adjacent frames — diagnosed
                            # in production as the cause of a genuine, fully
                            # content-independent false "Smoke Detected"
                            # alert flood after enough loop passes let the
                            # vote buffer accumulate hits from the seam alone
                            # (see FireSmokeDetector.reset_temporal_state's
                            # docstring for how this was isolated and
                            # confirmed). Only reset on an actual loop
                            # restart, never on ordinary frame reads.
                            try:
                                existing_session = inference_engine._sessions.get(info.camera_id)
                                if existing_session is not None:
                                    existing_session.fire_smoke_detector.reset_temporal_state()
                            except Exception:
                                pass
                            if not ret:
                                raise ConnectionError("Frame read failed.")
                    else:
                        # Live source — read via the watchdog queue (see
                        # `_threaded_reader` above) instead of calling
                        # cap.read() directly, so a driver stall can never
                        # freeze this loop: a timeout here is treated the
                        # same as a hard read failure, which the existing
                        # backoff-and-reopen logic below already recovers
                        # from by releasing and reconnecting.
                        try:
                            ret, frame = frame_q.get(timeout=5.0)
                        except _queue.Empty:
                            raise ConnectionError("Frame read timed out — camera appears stuck.")
                        if not ret:
                            raise ConnectionError("Frame read failed.")

                    # Resize for consistent streaming
                    h, w = frame.shape[:2]
                    target_w = 960
                    target_h = int(h * target_w / w)
                    frame = cv2.resize(frame, (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)

                    # Checked every frame so toggling "Enable AI Detection
                    # Processing" in Camera Settings takes effect instantly.
                    ai_on = self.is_ai_enabled(info.camera_id)

                    if ai_on:
                        # Render the latest available analysis onto THIS frame
                        # and push exactly one JPEG — the only writer to this
                        # buffer, so playback stays smooth regardless of AI speed.
                        with analysis_lock:
                            analysis = latest_analysis
                        # Live Monitoring / Cameras / Alerts etc. all read
                        # this buffer — behaviour-only overlay, structurally
                        # incapable of showing fire/smoke results even
                        # though `analysis` carries them (see the mandate to
                        # keep the two monitoring views fully separate).
                        annotated = inference_engine.render_behaviour_only(frame, analysis)
                        ok, jpeg = cv2.imencode(".jpg", annotated, encode_params)
                        if ok:
                            info.buffer.push(jpeg.tobytes())

                        if info.fire_smoke_buffer is not None:
                            fs_annotated = inference_engine.render_fire_smoke_only(frame, analysis, camera_name=str(info.camera_id))
                            ok_fs, jpeg_fs = cv2.imencode(".jpg", fs_annotated, encode_params)
                            if ok_fs:
                                jpeg_fs_bytes = jpeg_fs.tobytes()
                                info.fire_smoke_buffer.push(jpeg_fs_bytes)
                                self._update_fire_smoke_incidents(info, jpeg_fs_bytes)

                        # Hand the raw frame to the AI worker for the next analysis pass
                        try:
                            infer_q.put_nowait(frame.copy())
                        except _queue.Full:
                            try:
                                _ = infer_q.get_nowait()
                                infer_q.task_done()
                            except (_queue.Empty, Exception):
                                pass
                            try:
                                infer_q.put_nowait(frame.copy())
                            except (_queue.Full, Exception):
                                pass
                    else:
                        # AI disabled for this camera: stream the clean feed,
                        # no detection running, no boxes/labels drawn.
                        with analysis_lock:
                            latest_analysis = FrameAnalysis()
                        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
                        if ok:
                            info.buffer.push(jpeg.tobytes())
                            if info.fire_smoke_buffer is not None:
                                info.fire_smoke_buffer.push(jpeg.tobytes())

                    # Pace to source FPS only for video files to avoid artificial buffering lag on live streams
                    if is_video_file:
                        elapsed = time.time() - t0
                        sleep_t = frame_delay - elapsed
                        if sleep_t > 0:
                            time.sleep(sleep_t)

            except Exception as exc:
                info.consecutive_failures += 1
                info.error = str(exc)
                logger.warning(
                    f"Camera {info.camera_id} error (attempt {info.consecutive_failures}): {exc}"
                )
                reader_stop.set()
                if cap:
                    cap.release()
                if not info.stop_event.is_set():
                    time.sleep(min(backoff, max_backoff))
                    backoff *= 2
            finally:
                reader_stop.set()
                if cap:
                    cap.release()

        # Signal inference worker to exit
        infer_q.put(None)
        worker.join(timeout=3.0)
        logger.info(f"Camera {info.camera_id}: capture thread exited.")

    def _update_fire_smoke_incidents(self, info: "StreamInfo", jpeg_fs_bytes: Optional[bytes]) -> None:
        """
        Builds the PRE-EVENT + ACTIVE EVENT + POST-EVENT clip for a Fire/
        Smoke incident, called once per rendered frame from the capture
        loop (never blocking it — encoding happens on a short-lived
        background thread once an incident actually closes).

        State transitions come from `info.fire_smoke_incident_signals`,
        written by the AI worker thread (see `_inference_worker`) whenever
        it dispatches a new Fire/Smoke alert ("start") or sees that
        incident's "finalize"/"reopen" event-update ("end"/"start"). This
        method just consumes those signals and does the actual frame
        bookkeeping + encoding hand-off.
        """
        with info.fire_smoke_incident_signals_lock:
            signals = dict(info.fire_smoke_incident_signals)
            info.fire_smoke_incident_signals.clear()

        # No additional post-event grace is layered on top of fire_smoke_
        # state.py's own END_GRACE_SECONDS here — that grace already delays
        # "finalize" until confidence has genuinely stayed low for a while,
        # and this method keeps appending frames every tick up until
        # finalize actually fires, so the incident's "active" recording
        # already includes that closing-grace period as the clip's
        # post-event tail. Finalizing on a SEPARATE, later timer would only
        # add redundant delay before the fuller clip replaces the initial
        # snapshot (see the "update_clip" mechanism below).
        for label in FIRE_SMOKE_LABELS:
            sig = signals.get(label)
            incident = info.fire_smoke_incidents.get(label)

            if sig == "start" and incident is None:
                # New incident — seed with whatever's already in the
                # fire/smoke buffer (the PRE-EVENT window).
                info.fire_smoke_incidents[label] = {
                    "frames": deque(info.fire_smoke_buffer.recent(), maxlen=MAX_FIRE_SMOKE_INCIDENT_FRAMES),
                }
            elif sig == "end" and incident is not None:
                frames = list(incident["frames"])
                del info.fire_smoke_incidents[label]
                threading.Thread(
                    target=self._finalize_fire_smoke_clip,
                    args=(info.camera_id, label, frames),
                    daemon=True,
                    name=f"fs-clip-{info.camera_id}-{label}",
                ).start()

        for incident in info.fire_smoke_incidents.values():
            if jpeg_fs_bytes is not None:
                incident["frames"].append(jpeg_fs_bytes)

    @staticmethod
    def _finalize_fire_smoke_clip(camera_id: int, activity: str, jpeg_frames: list) -> None:
        """
        Encodes the full pre+active+post frame list into one verified clip
        (see ai_service/video_processor.py) and attaches it to the
        already-created (and by now likely already-closed) alert row via an
        "update_clip" event-update — runs on its own short-lived thread so
        encoding a potentially long clip never delays the capture loop.
        """
        try:
            from app.services import alert_dispatcher
            processor = alert_dispatcher._get_video_processor()
            clip_path = processor.save_clip_from_jpegs(jpeg_frames, camera_id, fps=15.0, prefix="firesmoke")
            thumbnail_path = None
            if jpeg_frames:
                import cv2
                import numpy as np
                first_frame = cv2.imdecode(np.frombuffer(jpeg_frames[0], np.uint8), cv2.IMREAD_COLOR)
                if first_frame is not None:
                    thumbnail_path = processor.save_snapshot(first_frame, camera_id, prefix="firesmoke")
            if clip_path or thumbnail_path:
                alert_dispatcher.dispatch_event_update_sync({
                    "action": "update_clip", "camera_id": camera_id, "activity": activity,
                    "video_clip_path": clip_path, "thumbnail_path": thumbnail_path,
                })
        except Exception as e:
            logger.warning(f"Camera {camera_id}: failed to finalize fire/smoke incident clip for '{activity}': {e}")

    @staticmethod
    def _build_capture_url(
        source_url: str,
        source_type: str,
        username: str = "",
        password: str = "",
    ) -> str:
        """Build OpenCV-compatible capture URL."""
        if source_type == "webcam":
            # source_url is the device index (e.g. "0")
            try:
                return int(source_url)
            except ValueError:
                return 0
        if source_type == "rtsp" and username and password:
            # Inject credentials into RTSP URL
            if "://" in source_url:
                scheme, rest = source_url.split("://", 1)
                return f"{scheme}://{username}:{password}@{rest}"
        return source_url

    @staticmethod
    def _placeholder_frame(camera_id: int) -> bytes:
        """Generate a small JPEG placeholder when no stream is available."""
        import numpy as np
        img = np.zeros((270, 480, 3), dtype="uint8")
        cv2.putText(
            img, f"Camera {camera_id}", (80, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2,
        )
        cv2.putText(
            img, "Connecting...", (115, 165),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1,
        )
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    @staticmethod
    def _blocking_test_connection(
        source_url: str,
        source_type: str,
        username: str,
        password: str,
        timeout: float,
    ) -> dict:
        """Synchronous connection test (called from executor)."""
        url = VideoStreamService._build_capture_url(source_url, source_type, username, password)
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
        try:
            if not cap.isOpened():
                return {"success": False, "error": "Cannot open stream", "fps": None, "resolution": None}
            ret, frame = cap.read()
            if not ret:
                return {"success": False, "error": "Frame read failed", "fps": None, "resolution": None}
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            return {
                "success": True,
                "error": None,
                "fps": fps,
                "resolution": f"{w}x{h}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "fps": None, "resolution": None}
        finally:
            cap.release()


# Global singleton â€” initialized once at application startup
video_stream_service = VideoStreamService()
