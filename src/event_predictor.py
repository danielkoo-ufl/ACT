import math
import datetime as dt

import joblib
import pandas as pd

# Must match the columns used in train_model.py
NUM_SCALE = ["age", "minute_of_day"]
NUM_LOG   = ["dwell_seconds"]
NUM_PASS  = ["hour_sin", "hour_cos"]
BINARY    = ["is_weekend", "is_school_holiday", "is_off_routine"]
CAT       = ["day_of_week", "occupation", "object", "action",
             "prev_action", "object_in_hand", "zone"]
INPUT_COLUMNS = NUM_SCALE + NUM_LOG + NUM_PASS + BINARY + CAT

# Verbs where the person is actually holding the object
HELD_VERBS = {"picking up", "holding", "drinking from", "using"}

# Vision verb -> the action vocabulary used in dataset.csv
# (training vocabulary: holding, picking_up, reaching, sitting, standing, walking, lying_down)
# Verbs the dataset has no equivalent for are mapped to the closest one: adjust as you see fit.
ACTION_MAP = {
    "lying down": "lying_down",
    "holding": "holding",
    "reaching for": "reaching",
    "sitting": "sitting",
    "walking": "walking",
    "standing": "standing",
    "picking up": "picking_up",
    "putting down": "putting_down",
    "using": "using",
    "none": "none",
}


OBJECT_MAP = {
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "bottle": "bottle",
    "cup": "cup",
    "cell phone": "cell phone",
    "book": "book",
    "pen": "pen",
    "pencil": "pencil",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "remote": "remote",
    "toothbrush": "toothbrush",
    "bowl": "bowl",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "chair": "chair",
    "couch": "couch",
    "bed": "bed",
}


NONE_ACTION = "standing"   # value for "no previous action"
NONE_OBJECT = "none"       # value for "no object visible"


def day_name(now):
    """day_of_week is an int in the training data. Python: Monday=0 ... Sunday=6."""
    return now.weekday()


def _py(x):
    return x.item() if hasattr(x, "item") else x


class EventPredictor:
    def __init__(self, model_path, context, start_time=None, skip_actions=("none",)):
        """
        model_path : path to event_model.joblib
        context    : dict with age, occupation, zone, is_school_holiday, is_off_routine
        start_time : datetime the video started at (recorded footage).
                     None => use the real clock (live webcam).
        """
        self.model = joblib.load(model_path)
        self.ctx = context
        self.start_time = start_time
        self.skip = set(skip_actions)
        self.prev_action = NONE_ACTION
        self.classes = self.model.named_steps["clf"].classes_

        ohe = self.model.named_steps["pre"].named_transformers_["ohe"]
        self.known = {c: [_py(v) for v in cats] for c, cats in zip(CAT, ohe.categories_)}
        self._warned = set()

        # Validate the static context up front
        for c in ("occupation", "zone"):
            if self.ctx[c] not in self.known[c]:
                raise ValueError(f"context['{c}']='{self.ctx[c]}' is not in the training data. "
                                 f"Choose one of: {self.known[c]}")

    # ------------------------------------------------------------ helpers
    def _clock(self, video_time_s):
        if self.start_time is None:
            return dt.datetime.now()
        return self.start_time + dt.timedelta(seconds=video_time_s)

    def _check(self, row):
        # OneHotEncoder(handle_unknown="ignore") silently zeros unseen categories,
        # so warn loudly instead.
        for c in CAT:
            key = (c, row[c])
            if row[c] not in self.known[c] and key not in self._warned:
                self._warned.add(key)
                print(f"[WARN] '{row[c]}' was never seen for '{c}' in training. "
                      f"Known: {self.known[c]}")

    # ------------------------------------------------------------ public API
    def build_row(self, verb, obj, dwell_s, video_time_s):
        now = self._clock(video_time_s)
        hour_frac = now.hour + now.minute / 60
        action = ACTION_MAP.get(verb, verb)
        obj = OBJECT_MAP.get(obj, obj) if obj else None
        holding = int(verb in HELD_VERBS and obj is not None)

        row = {
            "minute_of_day": now.hour * 60 + now.minute,
            "hour_sin": math.sin(2 * math.pi * hour_frac / 24),
            "hour_cos": math.cos(2 * math.pi * hour_frac / 24),
            "age": self.ctx["age"],
            "dwell_seconds": max(dwell_s, 0.0),
            "day_of_week": day_name(now),
            "is_weekend": int(now.weekday() >= 5),
            "is_school_holiday": int(self.ctx["is_school_holiday"]),
            "is_off_routine": int(self.ctx["is_off_routine"]),
            "occupation": self.ctx["occupation"],
            "zone": self.ctx["zone"],
            "object": obj or NONE_OBJECT,
            "action": action,
            "prev_action": self.prev_action,
            "object_in_hand": holding,          # 0/1 flag in the training data
        }
        self._check(row)
        return pd.DataFrame([row], columns=INPUT_COLUMNS)

    def predict(self, verb, obj, dwell_s, video_time_s):
        X = self.build_row(verb, obj, dwell_s, video_time_s)
        proba = self.model.predict_proba(X)[0]
        i = int(proba.argmax())
        return self.classes[i], float(proba[i]), dict(zip(self.classes, proba.round(3)))

    def close_segment(self, verb):
        a = ACTION_MAP.get(verb, verb)
        if a in self.known["action"]:        # ignore none/touching/etc.
            self.prev_action = a