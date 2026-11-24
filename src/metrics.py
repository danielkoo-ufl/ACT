
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Total event predictions, categorized by event class
PREDICTIONS = Counter(
    "act_predictions_total",
    "Total number of event predictions",
    ["event_class"]
)

# Time spent processing each frame
PROCESSING_TIME = Histogram(
    "act_processing_seconds",
    "Time spent processing a frame"
)

# Confidence of the latest event prediction
CONFIDENCE = Gauge(
    "act_prediction_confidence",
    "Latest event prediction confidence"
)


def start_monitoring():
    start_http_server(8000, addr="0.0.0.0")
    print("Prometheus metrics available at http://localhost:8000/metrics")
