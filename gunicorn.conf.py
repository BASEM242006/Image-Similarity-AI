"""Gunicorn configuration tuned for Render free tier (512 MB RAM, 0.1 CPU)."""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Keep a single worker so TensorFlow and the Siamese model are loaded once in RAM.
workers = 1
threads = 2
worker_class = "gthread"

# Allow sufficient time for TensorFlow model warmup and CPU inference on free tier.
timeout = 120
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"
