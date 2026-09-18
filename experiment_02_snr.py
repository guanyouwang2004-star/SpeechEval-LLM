"""Controlled background-noise experiment for SpeechEval-LLM.

The script adds deterministic white noise to one clean speech recording at
several target SNR values. Because the clean signal and injected noise are both
known, the experiment can verify the measured SNR before a rule-based or LLM
diagnostic layer is introduced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 48_000
EPSILON = 1e-12


def rms(signal: np.ndarray) -> float:
    """Return linear RMS amplitude."""
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def dbfs(value: float) -> float:
    """Convert a linear full-scale amplitude to dBFS."""
    return float(20.0 * np.log10(max(value, EPSILON)))


def add_white_noise_at_snr(
    clean: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a noisy copy and the exact noise signal at the requested SNR."""
    clean_rms = rms(clean)
    if clean_rms <= EPSILON:
        raise ValueError("Input speech is silent; SNR cannot be constructed.")

    raw_noise = rng.standard_normal(clean.shape).astype(np.float32)
    raw_noise_rms = rms(raw_noise)
    target_noise_rms = clean_rms / (10.0 ** (target_snr_db / 20.0))
    noise = raw_noise * (target_noise_rms / max(raw_noise_rms, EPSILON))
    noisy = clean + noise

    # Apply one common gain to preserve SNR while preventing saved WAV clipping.
    peak = float(np.max(np.abs(noisy)))
    if peak > 0.99:
        gain = 0.99 / peak
        noisy = noisy * gain
        noise = noise * gain
        clean_for_measurement = clean * gain
    else:
        clean_for_measurement = clean

    return (
        noisy.astype(np.float32),
        noise.astype(np.float32),
        clean_for_measurement.astype(np.float32),
    )


def measure_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Measure SNR when the clean reference and injected noise are known."""
    return float(20.0 * np.log10((rms(clean) + EPSILON) / (rms(noise) + EPSILON)))


def diagnose_snr(measured_snr_db: float) -> tuple[str, str, str]:
    """Return a transparent rule-baseline label, interpretation, and action."""
    # Stabilize exact experimental boundaries against floating-point noise.
    snr = round(float(measured_snr_db), 6)

    if snr >= 20.0:
        return (
            "low_noise_risk",
            "Background noise is unlikely to be the dominant intelligibility problem.",
            "Check reverberation, late reflections, frequency response, and alignment.",
        )
    if snr >= 10.0:
        return (
            "mild_noise_risk",
            "Noise may reduce consonant clarity for distant or quiet listeners.",
            "Measure the real room noise spectrum and improve source-to-noise ratio.",
        )
    if snr >= 5.0:
        return (
            "moderate_noise_risk",
            "Background noise is a plausible contributor to reduced intelligibility.",
            "Prioritize noise control, closer microphone placement, or level optimization.",
        )
    if snr >= 0.0:
        return (
            "severe_noise_risk",
            "Speech and noise have comparable energy, so masking is likely substantial.",
            "Reduce noise at the source before adding more loudspeaker level.",
        )
    return (
        "critical_noise_risk",
        "Noise energy exceeds speech energy and is likely to dominate perception.",
        "Treat the noise source and re-measure before evaluating other causes.",
    )


def spectral_centroid_hz(signal: np.ndarray, sample_rate: int) -> float:
    """Return mean spectral centroid as a secondary, non-diagnostic feature."""
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    return float(np.mean(centroid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled SpeechEval-LLM SNR experiments."
    )
    parser.add_argument("--input", default="data/test.wav", help="Clean speech WAV")
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[20.0, 10.0, 5.0, 0.0],
        help="Target SNR values in dB",
    )
    parser.add_argument("--seed", type=int, default=42, help="Noise random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_audio_dir = Path("data/generated")
    results_dir = Path("results")
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    clean, sample_rate = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    signals: list[tuple[str, np.ndarray]] = [("Clean reference", clean)]

    for target_snr_db in args.snrs:
        noisy, noise, clean_for_measurement = add_white_noise_at_snr(
            clean,
            target_snr_db,
            rng,
        )
        measured_snr = measure_snr_db(clean_for_measurement, noise)
        label, interpretation, intervention = diagnose_snr(measured_snr)

        snr_name = f"{target_snr_db:g}".replace("-", "minus_").replace(".", "p")
        output_path = output_audio_dir / f"test_snr_{snr_name}db.wav"
        sf.write(output_path, noisy, sample_rate)

        row = {
            "experiment": "SNR_Noise",
            "target_snr_db": round(float(target_snr_db), 2),
            "measured_snr_db": round(measured_snr, 2),
            "output_rms_dbfs": round(dbfs(rms(noisy)), 2),
            "output_peak_dbfs": round(dbfs(float(np.max(np.abs(noisy)))), 2),
            "spectral_centroid_hz": round(
                spectral_centroid_hz(noisy, sample_rate), 2
            ),
            "diagnosis": label,
            "interpretation": interpretation,
            "recommended_verification_or_action": intervention,
        }
        rows.append(row)
        signals.append((f"{target_snr_db:g} dB SNR", noisy))

    csv_path = results_dir / "experiment_02_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(len(signals), 1, figsize=(11, 2.6 * len(signals)))
    for axis, (title, signal) in zip(axes, signals):
        spectrogram = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal)),
            ref=np.max,
        )
        librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            x_axis="time",
            y_axis="log",
            ax=axis,
        )
        axis.set_title(title)
    figure.suptitle("Experiment 02 - Controlled White Noise / SNR", fontsize=14)
    figure.tight_layout()
    figure.savefig(results_dir / "experiment_02_spectrograms.png", dpi=160)
    plt.close(figure)

    print("====================================")
    print(" SpeechEval-LLM SNR Experiment")
    print("====================================")
    for row in rows:
        print(
            f"Target {row['target_snr_db']:>5} dB | "
            f"Measured {row['measured_snr_db']:>5} dB | "
            f"{row['diagnosis']}"
        )
    print(f"\nSaved table: {csv_path}")
    print("Saved plot: results/experiment_02_spectrograms.png")


if __name__ == "__main__":
    main()
