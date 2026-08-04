"""Generate the IEEE Figure 1 representative input-evidence figure.

This script is intentionally isolated from model-training, preprocessing,
inference, and application code. It uses one PalsyNet affected-class video,
one TORGO recording, and the project schemas to render the four evidence
panels described in the paper.

PalsyNet source: https://huggingface.co/datasets/jasir/palsynet-data
PalsyNet licence: CC BY 4.0
Audio settings mirror rural_stroke_assist.features.speech_features.load_audio:
mono, target sample rate 16,000 Hz, maximum display duration 5 seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "figures" / "input_evidence_examples"
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_DISPLAY_SECONDS = 5.0

PALSYNET_DATASET_NAME = "PalsyNet / Facial Palsy Dataset"
PALSYNET_DATASET_URL = "https://huggingface.co/datasets/jasir/palsynet-data"
PALSYNET_LICENSE = "CC BY 4.0"
PALSYNET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

FACE_BLUE = "#2f6f9f"
FACE_FILL = "#edf5fb"
SPEECH_GREEN = "#3e8e68"
SPEECH_FILL = "#edf7f1"
SYMPTOM_ORANGE = "#c87926"
SYMPTOM_FILL = "#fff5e8"
CONTEXT_PURPLE = "#7656a5"
CONTEXT_FILL = "#f5f0fb"
TEXT = "#20252b"
MUTED = "#59636e"
GRID = "#d8dde2"

SYMPTOM_FIELDS = [
    ("Face drooping", "face_drooping"),
    ("Arm weakness", "arm_weakness"),
    ("Speech difficulty", "speech_difficulty"),
    ("Balance / coordination loss", "balance_or_coordination_loss"),
    ("Vision disturbance", "vision_disturbance"),
]

SYNTHETIC_METADATA_VALUES = {
    "age": "67 years",
    "hypertension": "No",
    "heart_disease": "No",
    "avg_glucose_level": "103 mg/dL",
    "bmi": "27.4",
    "smoking_status": "Never",
}

METADATA_FIELDS = [
    ("Age", "age"),
    ("Hypertension", "hypertension"),
    ("Heart disease", "heart_disease"),
    ("Glucose", "avg_glucose_level"),
    ("BMI", "bmi"),
    ("Smoking status", "smoking_status"),
]


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a CLI path without relying on a user-specific absolute path."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def clip_audio_preview(
    signal: np.ndarray,
    sample_rate: int,
    max_duration_seconds: float = MAX_AUDIO_DISPLAY_SECONDS,
) -> np.ndarray:
    """Return a non-empty mono signal limited to the project display window."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if signal is None or signal.size == 0:
        raise ValueError("empty audio signal")
    max_samples = max(1, int(round(sample_rate * max_duration_seconds)))
    return np.asarray(signal).reshape(-1)[:max_samples]


def load_audio_preview(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load one TORGO file using the project speech preprocessing settings."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    signal, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_AUDIO_DISPLAY_SECONDS,
    )
    return clip_audio_preview(signal, sample_rate), sample_rate


def extract_face_frame(
    video_path: Path,
    output_path: Path,
    frame_fraction: float = 0.55,
) -> tuple[Path, int, int, float]:
    """Extract one frame without cropping, contrast adjustment, or resizing."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Face video does not exist: {video_path}")
    if not 0.0 <= frame_fraction <= 1.0:
        raise ValueError("frame_fraction must be between 0 and 1")

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Face video has no frames: {video_path}")

    frame_index = min(frame_count - 1, int(round(frame_count * frame_fraction)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.size == 0:
        raise ValueError(f"Could not read selected face frame: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
        raise OSError(f"Could not write extracted face frame: {output_path}")
    return output_path, frame_index, frame_count, fps


def load_face_image(face_image_path: Path) -> np.ndarray:
    """Load a face frame as RGB while preserving its original aspect ratio."""
    if not face_image_path.is_file():
        raise FileNotFoundError(f"Face image does not exist: {face_image_path}")
    with Image.open(face_image_path) as image:
        return np.asarray(image.convert("RGB"))


def style_axes(ax: mpl.axes.Axes, border_color: str, fill_color: str) -> None:
    ax.set_facecolor(fill_color)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(border_color)


def add_panel_heading(ax: mpl.axes.Axes, heading: str, subtitle: str) -> None:
    ax.text(
        0.055,
        0.955,
        heading,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        color=TEXT,
        linespacing=1.0,
    )
    ax.text(
        0.055,
        0.785,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=MUTED,
    )


def draw_face_panel(ax: mpl.axes.Axes, face_image: np.ndarray) -> None:
    style_axes(ax, FACE_BLUE, FACE_FILL)
    add_panel_heading(ax, "(a) Facial Appearance", "Public facial-palsy example")
    inset = ax.inset_axes([0.09, 0.095, 0.82, 0.62])
    inset.imshow(face_image, aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.035,
        "Proxy visual evidence",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=FACE_BLUE,
    )


def draw_speech_panel(ax: mpl.axes.Axes, signal: np.ndarray, sample_rate: int) -> None:
    style_axes(ax, SPEECH_GREEN, SPEECH_FILL)
    add_panel_heading(ax, "(b) Speech Recording", "TORGO dysarthria recording")

    waveform_ax = ax.inset_axes([0.095, 0.19, 0.84, 0.47])
    time = np.arange(signal.size, dtype=float) / sample_rate
    if signal.size > 3000:
        sample_indices = np.linspace(0, signal.size - 1, 3000).astype(int)
        time = time[sample_indices]
        signal = signal[sample_indices]
    amplitude = np.max(np.abs(signal)) or 1.0
    normalized = signal / amplitude
    waveform_ax.plot(time, normalized, color=SPEECH_GREEN, linewidth=0.45)
    waveform_ax.axhline(0.0, color=GRID, linewidth=0.45)
    waveform_ax.set_xlim(0.0, max(float(time[-1]), 0.1))
    waveform_ax.set_ylim(-1.05, 1.05)
    waveform_ax.set_xticks([0.0, round(float(time[-1]), 1)])
    waveform_ax.set_yticks([])
    waveform_ax.set_xlabel("time (s)", fontsize=5.4, color=MUTED, labelpad=1)
    waveform_ax.tick_params(axis="x", labelsize=5.1, colors=MUTED, length=2, pad=1)
    waveform_ax.set_facecolor("white")
    for spine in waveform_ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.5)
    ax.text(
        0.5,
        0.055,
        "16 kHz - first 5 s shown",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=SPEECH_GREEN,
    )


def draw_symptom_panel(ax: mpl.axes.Axes) -> None:
    style_axes(ax, SYMPTOM_ORANGE, SYMPTOM_FILL)
    add_panel_heading(ax, "(c) FAST / BE-FAST\nResponses", "Structured acute symptoms")
    row_y = np.linspace(0.66, 0.19, len(SYMPTOM_FIELDS))
    for (label, _field_name), y in zip(SYMPTOM_FIELDS, row_y):
        ax.add_patch(
            Rectangle(
                (0.085, y - 0.035),
                0.062,
                0.062,
                facecolor="white",
                edgecolor=SYMPTOM_ORANGE,
                linewidth=0.8,
            )
        )
        ax.text(
            0.18,
            y - 0.002,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color=TEXT,
        )
    ax.text(
        0.5,
        0.055,
        "Neutral example state",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=SYMPTOM_ORANGE,
    )


def draw_context_panel(ax: mpl.axes.Axes) -> None:
    style_axes(ax, CONTEXT_PURPLE, CONTEXT_FILL)
    add_panel_heading(ax, "(d) Patient Context", "Synthetic contextual fields")
    row_y = np.linspace(0.68, 0.19, len(METADATA_FIELDS))
    for (label, field_name), y in zip(METADATA_FIELDS, row_y):
        ax.text(
            0.085,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.75,
            color=TEXT,
        )
        ax.text(
            0.91,
            y,
            SYNTHETIC_METADATA_VALUES[field_name],
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.75,
            color=CONTEXT_PURPLE,
            fontweight="bold",
        )
        ax.plot([0.085, 0.915], [y - 0.055, y - 0.055], color=GRID, linewidth=0.45)
    ax.text(
        0.5,
        0.055,
        "All displayed values are synthetic",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=CONTEXT_PURPLE,
    )


def generate_figure(
    face_image_path: Path,
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Path | int | float]:
    """Render and export the single four-panel figure."""
    face_image = load_face_image(face_image_path)
    signal, sample_rate = load_audio_preview(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(3.8, 4.8), facecolor="white")
    grid = GridSpec(
        2,
        2,
        figure=figure,
        left=0.055,
        right=0.945,
        bottom=0.04,
        top=0.965,
        wspace=0.12,
        hspace=0.16,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    draw_face_panel(axes[0], face_image)
    draw_speech_panel(axes[1], signal, sample_rate)
    draw_symptom_panel(axes[2])
    draw_context_panel(axes[3])

    pdf_path = output_dir / "input_evidence_examples.pdf"
    png_path = output_dir / "input_evidence_examples.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(figure)
    return {
        "pdf_path": pdf_path,
        "png_path": png_path,
        "face_width": int(face_image.shape[1]),
        "face_height": int(face_image.shape[0]),
        "audio_sample_rate": int(sample_rate),
        "audio_samples_displayed": int(signal.size),
        "audio_duration_displayed": float(signal.size / sample_rate),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-video", help="PalsyNet affected-class source video")
    parser.add_argument("--face-image", help="Existing extracted face frame")
    parser.add_argument("--audio-file", required=True, help="Readable TORGO audio file")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the extracted frame and figure outputs",
    )
    parser.add_argument(
        "--frame-fraction",
        type=float,
        default=0.55,
        help="Frame position as a fraction of the selected source video",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.face_video and not args.face_image:
        raise SystemExit("Provide --face-video or --face-image")

    output_dir = resolve_repo_path(args.output_dir)
    audio_path = resolve_repo_path(args.audio_file)
    face_video_path = resolve_repo_path(args.face_video) if args.face_video else None
    face_image_path = resolve_repo_path(args.face_image) if args.face_image else None
    extracted_frame_path = output_dir / "palsynet_affected_frame.jpg"

    frame_index = None
    frame_count = None
    fps = None
    if face_video_path is not None:
        face_image_path, frame_index, frame_count, fps = extract_face_frame(
            face_video_path,
            extracted_frame_path,
            frame_fraction=args.frame_fraction,
        )
    assert face_image_path is not None
    result = generate_figure(face_image_path, audio_path, output_dir)
    print(f"Face frame: {face_image_path}")
    if frame_index is not None:
        print(f"Face frame index: {frame_index}/{frame_count} at {fps:.2f} fps")
    print(f"Audio: {audio_path}")
    print(f"Audio sample rate: {result['audio_sample_rate']} Hz")
    print(f"Audio duration displayed: {result['audio_duration_displayed']:.3f} s")
    print(f"PDF: {result['pdf_path']}")
    print(f"PNG: {result['png_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
