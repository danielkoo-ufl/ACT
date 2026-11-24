import pandas as pd


# Returns the most recent event.

def latest_event(file_path, min_conf=0.5, min_dur=1.0, debug=False):
    ev = pd.read_csv(file_path, skipinitialspace=True)
    ev.columns = ev.columns.str.strip()
    if "pred_event" in ev.columns:
        ev["pred_event"] = ev["pred_event"].astype("string").str.strip()

    if debug:
        print("columns:", list(ev.columns))
        print("rows:", len(ev))

    if not {"pred_event", "pred_conf", "dur_s"}.issubset(ev.columns):
        return None, 0.0

    valid = ev.dropna(subset=["pred_event", "pred_conf"])
    valid = valid[(valid.pred_event != "none")
                  & (valid.pred_conf >= min_conf)
                  & (valid.dur_s >= min_dur)]

    if debug:
        print(valid[["label", "dur_s", "pred_event", "pred_conf"]])

    if valid.empty:
        return None, 0.0

    latest = valid.iloc[-1]
    return str(latest.pred_event), float(latest.pred_conf)

print(latest_event("events.csv", debug=False))
