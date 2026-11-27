"""Flask web application for Siamese Network V4 Image Similarity Detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from model_service import InvalidImageError, SiameseSimilarityService

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "siamese_v4.keras"
THRESHOLD_PATH = BASE_DIR / "similarity_threshold.npy"
MAX_UPLOAD_MB = 16


def create_app(service: SiameseSimilarityService | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    model_service = service or SiameseSimilarityService(MODEL_PATH, THRESHOLD_PATH)
    app.extensions["siamese_service"] = model_service

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            threshold=round(model_service.threshold, 4),
            max_upload_mb=MAX_UPLOAD_MB,
        )

    @app.get("/api/health")
    @app.get("/health")
    def health() -> tuple[Any, int]:
        if not model_service.is_ready():
            return jsonify({"status": "unavailable"}), 503
        return (
            jsonify(
                {
                    "status": "ok",
                    "model": "Siamese Network V4",
                    "model_file": model_service.model_path.name,
                    "threshold": round(model_service.threshold, 6),
                    "input_shape": [32, 32, 3],
                }
            ),
            200,
        )

    @app.post("/api/compare")
    def compare_images() -> tuple[Any, int]:
        if "image1" not in request.files or "image2" not in request.files:
            return (
                jsonify({"error": "Both 'image1' and 'image2' files are required."}),
                400,
            )

        file1 = request.files["image1"]
        file2 = request.files["image2"]

        if not file1.filename or not file2.filename:
            return (
                jsonify({"error": "Please select two valid image files before comparing."}),
                400,
            )

        try:
            bytes1 = file1.read()
            bytes2 = file2.read()
            result = model_service.compare_bytes(bytes1, bytes2)
        except InvalidImageError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            app.logger.exception("Unexpected error during Siamese model inference")
            return (
                jsonify({"error": "An internal error occurred while analyzing the images."}),
                500,
            )

        return jsonify(result.to_dict()), 200

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err: RequestEntityTooLarge) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "error": f"Uploaded payload exceeds the {MAX_UPLOAD_MB} MB limit. Please choose smaller images."
                }
            ),
            413,
        )

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("PORT", "5000"))
    application.run(host="0.0.0.0", port=port, debug=False)
