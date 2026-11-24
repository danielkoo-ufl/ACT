import os

import tomllib
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # ACT/
CONFIG_FILE = Path(__file__).resolve().parent / "config.toml"
ENV_FILE = ROOT / ".env"

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def _load_env(path):
    """Tiny .env reader. Real environment variables take priority."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load(path=None):
    _load_env(ROOT / ".env")
    path = Path(path or os.environ.get("ACT_CONFIG", ROOT / "config.toml"))
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    # resolve paths against the project folder, not the launch directory
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["log_file"]).parent.mkdir(parents=True, exist_ok=True)

    # secrets come from the environment only
    cfg["bulb"]["username"] = os.environ.get("TPLINK_USER", "")
    cfg["bulb"]["password"] = os.environ.get("TPLINK_PASS", "")

    # "" -> None (real clock); otherwise a datetime
    st = cfg["context"].pop("start_time", "")
    cfg["start_time"] = dt.datetime.fromisoformat(st) if st else None

    _validate(cfg)
    return cfg


def _validate(cfg):
    errors = []
    rt = cfg["runtime"]
    if rt["input"] not in ("webcam", "video", "stream"):
        errors.append(
            "runtime.input must be 'webcam', 'video', or 'stream'"
        )

    needed = ["yolo_model", "pose_model", "event_model"]
    if rt["input"] == "video":
        needed.append("video_file")
    for k in needed:
        if not Path(cfg["paths"][k]).exists():
            errors.append(f"paths.{k} not found: {cfg['paths'][k]}")

    if rt["input"] == "webcam" and cfg["start_time"] is not None:
        errors.append("context.start_time is set but input is a webcam; "
                      "leave it empty so the real clock is used")

    if rt["input"] == "stream" and not rt.get("stream_url"):
        errors.append("runtime.stream_url is required for stream input")
    b = cfg["bulb"]
    if b["enabled"]:
        if not (b["username"] and b["password"]):
            errors.append("bulb enabled but TPLINK_USER / TPLINK_PASS are not set (.env)")
        if not b["scenes"]:
            errors.append("bulb.scenes is empty")

    if errors:
        raise SystemExit("Config errors:\n  - " + "\n  - ".join(errors))