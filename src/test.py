import datetime as dt
from event_predictor import EventPredictor

ctx = dict(age=23, occupation="student_grad", zone="bed",
           is_school_holiday=0, is_off_routine=0)

for hour in (18, 21, 23):
    p = EventPredictor("artifacts/event_model.joblib", ctx,
                       start_time=dt.datetime(2026, 11, 10, hour, 30))
    for obj in (None, "bed"):
        print(hour, obj, p.predict("lying down", obj, 60, 0)[2])