import os, sys, signal, tempfile, threading, time, math, collections, logging
import datetime as dt
from logging.handlers import RotatingFileHandler
from pathlib import Path
import config
import cv2, numpy as np, pandas as pd, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from bulb import BulbController
from event_predictor import EventPredictor
from metrics import (
    PREDICTIONS,
    PROCESSING_TIME,
    CONFIDENCE,
    start_monitoring
)
start_monitoring()
log = logging.getLogger("act")
HEARTBEAT = Path(tempfile.gettempdir()) / "act_heartbeat"   # /tmp/act_heartbeat in the container
STOP = False                                                # set by SIGTERM (Kubernetes) for a clean exit

# ---------- what the system knows about (COCO class names) ----------

PORTABLE = {"backpack", "handbag", "suitcase", "bottle", "cup",
            "cell phone", "book", "pen", "pencil", "laptop",
            "keyboard", "mouse", "remote",
            "toothbrush", "bowl", "fork",
            "knife", "spoon"}

SEATS = {"chair", "couch", "bed"}

FACE_USE = {"cup": "drinking from",
            "bottle": "drinking from",
            "cell phone": "using",
            "fork": "using",
            "knife": "using",
            "spoon": "using",
            "toothbrush": "using",
            "laptop": "using",
            "keyboard": "using",
            "mouse": "using"}

SUGGESTIONS = {
    "picking up":    "Log pickup event",
    "holding":       "Object in use / being carried",
    "putting down":  "Log object released",
    "reaching for":  "Person about to interact",
    "touching":      "Contact with object",
    "drinking from": "Hydration detected",
    "using":         "Device in use",
    "sitting":       "Person seated",
    "lying down":    "Person lying down",
    "":              "",
}

# ---------- tunable settings (overridden by [vision] in config.toml) ----------
YOLO_CONF       = 0.3
NEAR_FRAC       = 0.35   # "near" = wrist within 35% of box's larger side
RISE_FRAC       = 0.04   # object must rise 4% of frame height while touched
RELEASE_GAP     = 5      # frames without a nearby wrist before "released"
PICK_SHOW       = 15     # frames to keep showing "picking up"
PUT_SHOW        = 15     # frames to keep showing "putting down"
REACH_FRAMES    = 3      # near for fewer frames than this = "reaching"
FACE_FRAC       = 0.20   # object within 20% of frame height of mouth
SIT_KNEE_MARGIN = 0.10
SMOOTH          = 3
LYING_ANGLE     = 35
LYING_HOLD_S    = 1.5    # keep "lying" true this long after the last lying reading
WRISTS, HIPS, KNEES, MOUTH, SHOULDER = (15, 16), (23, 24), (25, 26), (9, 10), (11, 12)

POSE_CONNECTION = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


# ---------- per-object state ----------

class ObjState:
    def __init__(self):
        self.base = None
        self.held = False
        self.gap = 0
        self.near_count = 0
        self.pick_until = -1
        self.put_until = -1
        self.last_seen = 0


def point_box_dist(px, py, box):
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - px, 0, px - x2), max(y1 - py, 0, py - y2))


def vis_ok(lm, t=0.4):
    return getattr(lm, "visibility", 1.0) > t


def make_landmarker(model_asset_path):
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_asset_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def _open_stream(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LiveCapture:
    """Reads a webcam / RTSP stream on a background thread, keeps only the newest
    frame (so latency never builds up when inference is slower than the camera),
    and reconnects by itself if the source drops."""

    def __init__(self, opener, reconnect_s=5):
        self.opener, self.reconnect_s = opener, reconnect_s
        self.cap = None
        self.frame, self.seq, self._seen = None, 0, 0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            if self.cap is None:
                cap = self.opener()
                if not cap.isOpened():
                    cap.release()
                    log.warning("input not available, retrying in %ss", self.reconnect_s)
                    time.sleep(self.reconnect_s)
                    continue
                log.info("input opened")
                self.cap = cap
            ok, frame = self.cap.read()
            if not ok:
                log.warning("input dropped, reconnecting in %ss", self.reconnect_s)
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_s)
                continue
            with self.lock:
                self.frame, self.seq = frame, self.seq + 1

    def read(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.running and time.monotonic() < end:
            with self.lock:
                if self.seq != self._seen:
                    self._seen = self.seq
                    return True, self.frame
            time.sleep(0.005)
        return False, None

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


# input_options: 0 = video file, 1 = live (webcam or stream)
# source       : stream URL (e.g. rtsp://...) used when input_options=1; None = webcam
# predictor    : an EventPredictor, or None to run the vision code only
# keep_records : store one dict per frame (only needed for frames.csv; costs memory)
def main(video_path, save_path=None, mirror=False, display_width=960,
         input_options=None, camera_index=0, predictor=None, bulb=None,
         yolo_path="YOLO/yolov8s.pt",
         pose_path="POSES/pose_landmarker_heavy.task",
         headless=False, source=None, reconnect_s=5, keep_records=True):
    pose_landmarker = make_landmarker(pose_path)
    yolo = YOLO(yolo_path)
    CLASSES = yolo.names
    if isinstance(CLASSES, list):
        CLASSES = dict(enumerate(CLASSES))

    OBJECT_CLASS_IDS = [cid for cid, cname in CLASSES.items() if cname is not None]

    live = input_options == 1
    cap = grabber = None
    if input_options == 0:
        if video_path is None:
            pose_landmarker.close()
            raise ValueError("video_path is required when input_options=0")
        cap = cv2.VideoCapture(video_path)
        source_name = video_path
        if not cap.isOpened():
            cap.release()
            pose_landmarker.close()
            raise RuntimeError(f"Cannot open input: {source_name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        read_frame = cap.read
    elif live:
        if source:
            source_name = "stream"                       # don't log the URL, it may hold credentials
            opener = lambda: _open_stream(source)
        else:
            source_name = f"Webcam {camera_index}"
            opener = lambda: cv2.VideoCapture(camera_index)
        grabber = LiveCapture(opener, reconnect_s)
        fps, total = 30.0, 0                             # fps only matters for video files
        read_frame = grabber.read
    else:
        pose_landmarker.close()
        raise ValueError("input_options must be 0 (video) or 1 (live)")

    log.info("input: %s (%s)", source_name, "live" if live else f"{total} frames, {fps:.1f} fps")

    writer = None
    draw = (not headless) or bool(save_path)             # skip drawing when nobody sees it

    states = {}
    votes = collections.deque(maxlen=SMOOTH)
    last_ts = -1
    frame_index = 0

    records = []                                         # one dict per frame (if keep_records)
    events = collections.deque(maxlen=5000)              # one dict per action segment
    cur = None                                           # current segment
    last_pred_text = ""                                  # prediction overlay that stays on screen
    last_logged = None
    pred_every = max(1, int(round(fps)))                 # video files: model about once per second

    t_start = time.monotonic()
    t_now = 0.0
    last_pred_m = 0.0
    last_beat = 0.0
    last_lying_t = -1e9

    def close_segment(seg, end_s):
        """Finish a segment, run the model on it, and store it as an event."""
        dwell = round(end_s - seg["start"], 2)
        ev = dict(label=seg["label"], start_s=seg["start"], end_s=end_s, dur_s=dwell)
        if predictor and seg["verb"] not in predictor.skip:

            with PROCESSING_TIME.time():
                pred, conf, _ = predictor.predict(
                    seg["verb"], seg["obj"], dwell, seg["start"]
                )

            PREDICTIONS.labels(event_class=str(pred)).inc()
            CONFIDENCE.set(float(conf))

            ev.update(pred_event=pred, pred_conf=round(conf, 3))


        if predictor:
            predictor.close_segment(seg["verb"])
        events.append(ev)

    try:
        while not STOP:
            # heartbeat for the Kubernetes liveness probe (touched even if the camera is down)
            now_m = time.monotonic()
            if now_m - last_beat >= 1.0:
                last_beat = now_m
                try:
                    HEARTBEAT.touch()
                except OSError:
                    pass

            ok, frame = read_frame()
            if not ok:
                if live:
                    continue                             # LiveCapture is reconnecting
                break                                    # end of video
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1

            # one clock for everything: wall clock when live, frame number for video
            t_now = (round(time.monotonic() - t_start, 3) if live
                     else round((frame_index - 1) / fps, 3))
            timestamp_ms = max(int(t_now * 1000), last_ts + 1)
            last_ts = timestamp_ms

            res = pose_landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

            # ---- 1. YOLO detection + tracking ----
            tracking_results = yolo.track(frame, persist=True, conf=YOLO_CONF,
                                          classes=OBJECT_CLASS_IDS, verbose=False)[0]
            objs = []                                    # (id, name, box)

            if (tracking_results.boxes is not None
                    and tracking_results.boxes.id is not None):
                for detection_box, track_id in zip(
                        tracking_results.boxes,
                        tracking_results.boxes.id.int().tolist()):
                    class_id = int(detection_box.cls[0])
                    objs.append((track_id, CLASSES[class_id],
                                 detection_box.xyxy[0].tolist()))

            # ---- 2. Pose ----
            wrists, mouth, hip_pt, knee_y, shoulder_pt = [], None, None, None, None
            if res.pose_landmarks:
                pose = res.pose_landmarks[0]             # single person
                if draw:
                    for a, b in POSE_CONNECTION:
                        cv2.line(frame, (int(pose[a].x * w), int(pose[a].y * h)),
                                        (int(pose[b].x * w), int(pose[b].y * h)), (255, 255, 0), 2)
                    for lm in pose:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

                wrists = [(pose[i].x * w, pose[i].y * h) for i in WRISTS if vis_ok(pose[i])]
                mouth = (np.mean([pose[i].x for i in MOUTH]) * w,
                         np.mean([pose[i].y for i in MOUTH]) * h)
                if all(vis_ok(pose[i]) for i in HIPS):
                    hip_pt = (np.mean([pose[i].x for i in HIPS]) * w,
                              np.mean([pose[i].y for i in HIPS]) * h)
                if all(vis_ok(pose[i]) for i in KNEES):
                    knee_y = np.mean([pose[i].y for i in KNEES]) * h
                if all(vis_ok(pose[i]) for i in SHOULDER):
                    shoulder_pt = (np.mean([pose[i].x for i in SHOULDER]) * w,
                                   np.mean([pose[i].y for i in SHOULDER]) * h)

            # ---- 3. Lying / Sitting ----
            raw_lying = False
            if shoulder_pt and hip_pt:
                dx = abs(shoulder_pt[0] - hip_pt[0])
                dy = abs(shoulder_pt[1] - hip_pt[1])
                raw_lying = math.degrees(math.atan2(dy, dx)) < LYING_ANGLE
            if raw_lying:
                last_lying_t = t_now
            # hysteresis: stay "lying" briefly after the last reading, so flicker
            # and brief pose loss don't chop the segment into pieces
            lying = raw_lying or (t_now - last_lying_t) < LYING_HOLD_S

            sitting = False
            if hip_pt and not lying:
                for _, name, (x1, y1, x2, y2) in objs:
                    if name in SEATS and x1 < hip_pt[0] < x2 and y1 < hip_pt[1] < y2:
                        if knee_y is None or abs(knee_y - hip_pt[1]) < SIT_KNEE_MARGIN * h:
                            sitting = True

            # ---- 4. Per-object interaction state machine ----
            candidates = []                              # (priority, verb, text)
            for tracking_id, name, box in objs:
                x1, y1, x2, y2 = box
                if draw:
                    color = (0, 165, 255) if name in PORTABLE else (200, 200, 200)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{name} #{tracking_id}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if name not in PORTABLE:
                    continue

                object_state = states.setdefault(tracking_id, ObjState())
                object_state.last_seen = frame_index
                obj_center_y = (y1 + y2) / 2

                near_px = NEAR_FRAC * max(x2 - x1, y2 - y1)
                near = bool(wrists) and min(point_box_dist(x, y, box) for x, y in wrists) < near_px

                if near:
                    object_state.gap = 0
                    object_state.near_count += 1
                    if object_state.base is None:
                        object_state.base = obj_center_y
                    if not object_state.held and (object_state.base - obj_center_y) > RISE_FRAC * h:
                        object_state.held = True
                        object_state.pick_until = frame_index + PICK_SHOW
                else:
                    object_state.gap += 1
                    object_state.near_count = 0
                    if object_state.gap > RELEASE_GAP:
                        if object_state.held:
                            object_state.put_until = frame_index + PUT_SHOW
                        object_state.held = False
                        object_state.base = None

                if object_state.held and frame_index < object_state.pick_until:
                    candidates.append((6, "picking up", f"picking up {name}"))

                elif object_state.held:
                    near_face = mouth and point_box_dist(*mouth, box) < FACE_FRAC * h
                    if near_face and name in FACE_USE:
                        v = FACE_USE[name]
                        candidates.append((4, v, f"{v} {name}"))
                    else:
                        candidates.append((3, "holding", f"holding {name}"))

                elif frame_index < object_state.put_until:
                    candidates.append((5, "putting down", f"putting down {name}"))

                elif near and name == "laptop" and sitting:
                    candidates.append((4, "using", "using laptop"))

                elif near:
                    v = "reaching for" if object_state.near_count < REACH_FRAMES else "touching"
                    candidates.append((2, v, f"{v} {name}"))

            if sitting:
                candidates.append((1, "sitting", "sitting"))
            if lying:
                candidates.append((1, "lying down", "lying down"))

            for t in [t for t, s in states.items() if frame_index - s.last_seen > 60]:
                del states[t]                            # forget objects gone ~2 s

            # ---- 5. Smooth + record ----
            best = max(candidates, key=lambda c: c[0]) if candidates else (0, "", "none")
            votes.append(best[1:])
            (verb, text), _ = collections.Counter(votes).most_common(1)[0]

            # object involved in the winning action (handles "cell phone" etc.)
            obj_name = max((n for _, n, _ in objs if text.endswith(n)),
                           key=len, default=None)

            if keep_records:
                records.append(dict(
                    frame=frame_index,
                    time_s=t_now,
                    action=verb or "none",
                    label=text,
                    object=obj_name,
                    raw_label=best[2],
                    objects_visible=[(tid, n) for tid, n, _ in objs],
                    n_wrists=len(wrists),
                    sitting=sitting,
                    lying=lying,
                ))

            # ---- 6. Segment tracking + model prediction ----
            if cur is None or cur["label"] != text:
                if cur is not None:
                    close_segment(cur, t_now)
                cur = dict(label=text, verb=verb or "none", obj=obj_name, start=t_now)

            # running prediction on the in-progress segment, about once per second
            if predictor and cur["verb"] not in predictor.skip:
                now_m = time.monotonic()
                due = (now_m - last_pred_m >= 1.0) if live else (frame_index % pred_every == 0)
                if due:
                    last_pred_m = now_m
                    with PROCESSING_TIME.time():
                        pred, conf, _ = predictor.predict(
                            cur["verb"],
                            cur["obj"],
                            t_now - cur["start"],
                            t_now
                        )

                    PREDICTIONS.labels(event_class=str(pred)).inc()
                    CONFIDENCE.set(float(conf))
                    last_pred_text = f"Predicted event: {pred} ({conf:.0%})"
                    if pred != last_logged:
                        log.info("action=%s  predicted=%s (%.0f%%)", text, pred, conf * 100)
                        last_logged = pred
                    if bulb:
                        bulb.update(pred, conf)
            elif bulb:
                bulb.update("none", 0.0)                 # no usable action: reset the streak

            # ---- 7. Overlay + display ----
            if draw:
                cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"Action: {text}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, SUGGESTIONS.get(verb, ""), (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"frame {frame_index}/{total}", (w - 170, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if last_pred_text:
                    cv2.putText(frame, last_pred_text, (10, h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if save_path:
                if writer is None:
                    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (w, h))
                writer.write(frame)

            if not headless:
                disp = frame
                if display_width and w > display_width:
                    disp = cv2.resize(frame, (display_width, int(h * display_width / w)))
                cv2.imshow("Action detector", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        if grabber:
            grabber.release()
        if cap:
            cap.release()
        if writer:
            writer.release()
        pose_landmarker.close()
        if not headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if cur is not None:
        close_segment(cur, t_now)
    return records, list(events)


def _setup_logging(log_file):
    handlers = [logging.StreamHandler(sys.stdout)]       # stdout -> `kubectl logs`
    if log_file:
        handlers.append(RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s")


def _request_stop(signum, frame):
    global STOP
    STOP = True


if __name__ == "__main__":
    cfg = config.load()
    _setup_logging(cfg["paths"].get("log_file"))
    signal.signal(signal.SIGTERM, _request_stop)         # Kubernetes sends SIGTERM on stop

    for k, v in cfg["vision"].items():                   # override the tunable constants
        globals()[k.upper()] = v

    rt, p, b = cfg["runtime"], cfg["paths"], cfg["bulb"]
    log.info("starting: input=%s start_time=%s local time=%s",
             rt["input"], cfg["start_time"], dt.datetime.now().isoformat(timespec="seconds"))

    predictor = EventPredictor(p["event_model"], cfg["context"],
                               start_time=cfg["start_time"])
    bulb = (BulbController(b["host"], b["username"], b["password"], b["scenes"],
                           b["min_conf"], b["hold"], b["cooldown"])
            if b["enabled"] else None)

    records, events = main(
        p["video_file"],
        save_path=f"{p['output_dir']}/annotated.mp4" if rt["save_video"] else None,
        mirror=rt["mirror"], display_width=rt["display_width"],
        input_options=0 if rt["input"] == "video" else 1,
        camera_index=rt["camera_index"],
        predictor=predictor, bulb=bulb,
        yolo_path=p["yolo_model"], pose_path=p["pose_model"],
        headless=rt["headless"],
        source=rt["stream_url"] if rt["input"] == "stream" else None,
        reconnect_s=rt["reconnect_s"],
        keep_records=rt["save_csv"],
    )

    if rt["save_csv"]:
        pd.DataFrame(records).to_csv(f"{p['output_dir']}/frames.csv", index=False)
        pd.DataFrame(events).to_csv(f"{p['output_dir']}/events.csv", index=False)
