from pathlib import Path

import cv2
from PIL import Image


def inspect_video(
    video_path,
    engine,
    sample_every=10,
    max_samples=120,
    progress_callback=None,
):
    """
    Analyze sampled frames from a video using VisionInspect.

    Parameters
    ----------
    video_path:
        Path to the video file.

    engine:
        VisionInspectInference instance.

    sample_every:
        Analyze every Nth frame.

    max_samples:
        Maximum number of sampled frames to analyze.

    progress_callback:
        Optional function receiving progress from 0.0 to 1.0.

    Returns
    -------
    Dictionary containing:
        fps
        total_frames
        duration_seconds
        analyzed_frames
        anomalous_frames
        anomaly_percentage
        max_score
        overall_prediction
        records
        best_frame
    """

    video_path = Path(
        video_path
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration_seconds = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    records = []

    best_frame = None
    best_score = float(
        "-inf"
    )

    frame_index = 0
    analyzed_frames = 0

    try:

        while True:

            success, frame = (
                capture.read()
            )

            if not success:
                break

            # Only analyze selected frames
            if (
                frame_index
                % sample_every
                != 0
            ):
                frame_index += 1
                continue

            # OpenCV BGR -> RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            pil_image = Image.fromarray(
                rgb_frame
            )

            # Run PatchCore
            result = engine.predict_image(
                pil_image
            )

            score = float(
                result["score"]
            )

            threshold = float(
                result["threshold"]
            )

            prediction = (
                "ANOMALOUS"
                if score >= threshold
                else "NORMAL"
            )

            timestamp = (
                frame_index / fps
            )

            records.append(
                {
                    "frame":
                        frame_index,

                    "time_seconds":
                        timestamp,

                    "score":
                        score,

                    "threshold":
                        threshold,

                    "prediction":
                        prediction,
                }
            )

            analyzed_frames += 1

            # Keep the most anomalous frame
            if score > best_score:

                best_score = score

                best_frame = {
                    "frame":
                        frame_index,

                    "time_seconds":
                        timestamp,

                    "score":
                        score,

                    "threshold":
                        threshold,

                    "prediction":
                        prediction,

                    "image":
                        result["image"],

                    "heatmap":
                        result["heatmap"],
                }

            # Update progress
            if (
                progress_callback
                is not None
            ):

                if total_frames > 0:

                    progress = (
                        frame_index
                        / total_frames
                    )

                else:

                    progress = (
                        analyzed_frames
                        / max_samples
                    )

                progress_callback(
                    min(
                        progress,
                        1.0,
                    )
                )

            # Stop extremely long videos
            if (
                analyzed_frames
                >= max_samples
            ):
                break

            frame_index += 1

    finally:

        capture.release()

    if not records:

        raise ValueError(
            "No frames could be analyzed from the video."
        )

    anomalous_frames = sum(
        record["prediction"]
        == "ANOMALOUS"
        for record in records
    )

    anomaly_percentage = (
        anomalous_frames
        / analyzed_frames
        * 100
    )

    overall_prediction = (
        "ANOMALOUS"
        if anomalous_frames > 0
        else "NORMAL"
    )

    if progress_callback is not None:
        progress_callback(
            1.0
        )

    return {
        "fps":
            fps,

        "total_frames":
            total_frames,

        "duration_seconds":
            duration_seconds,

        "sample_every":
            sample_every,

        "analyzed_frames":
            analyzed_frames,

        "anomalous_frames":
            anomalous_frames,

        "anomaly_percentage":
            anomaly_percentage,

        "max_score":
            best_score,

        "overall_prediction":
            overall_prediction,

        "records":
            records,

        "best_frame":
            best_frame,
    }