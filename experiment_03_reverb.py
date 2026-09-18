"""Controlled reverberation experiment for SpeechEval-LLM.

The script creates deterministic synthetic room impulse responses (RIRs),
convolves them with clean speech, and calculates EDT, T20-derived RT60, C50,
and D50 from the known RIR. Metrics are deliberately calculated from the RIR,
not guessed from the reverberant speech waveform.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 48_000
EPSILON = 1e-12


def generate_synthetic_rir(
    sample_rate: int,
    target_rt60_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a repeatable direct sound, early reflections, and diffuse tail."""
    if target_rt60_s <= 0:
        raise ValueError("RT60 must be greater than zero.")

    duration_s = max(1.0, target_rt60_s * 1.35)
    sample_count = int(np.ceil(duration_s * sample_rate))
    time_s = np.arange(sample_count, dtype=np.float64) / sample_rate

    # An amplitude envelope of exp(-6.9078 t / RT60) falls by 60 dB at RT60.
    envelope = np.exp(-6.907755 * time_s / target_rt60_s)
    diffuse = rng.standard_normal(sample_count) * envelope

    # Limit the synthetic late field to a speech-relevant acoustic bandwidth.
    sos = butter(
        4,
        [100.0, min(10_000.0, sample_rate * 0.45)],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    diffuse = sosfilt(sos, diffuse)

    # Fade in the diffuse field after the direct arrival.
    fade_samples = max(1, int(0.005 * sample_rate))
    diffuse[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)

    rir = 0.12 * diffuse
    rir[0] += 1.0

    early_reflections = (
        (0.012, 0.42),
        (0.024, -0.31),
        (0.039, 0.23),
    )
    for delay_s, amplitude in early_reflections:
        index = int(round(delay_s * sample_rate))
        if index < sample_count:
            rir[index] += amplitude * np.exp(-6.907755 * delay_s / target_rt60_s)

    peak = float(np.max(np.abs(rir)))
    return (rir / max(peak, EPSILON)).astype(np.float32)


def energy_decay_curve_db(rir: np.ndarray) -> np.ndarray:
    """Calculate the normalized Schroeder backward-integrated decay curve."""
    energy = np.square(rir.astype(np.float64))
    decay = np.cumsum(energy[::-1])[::-1]
    decay /= max(float(decay[0]), EPSILON)
    return 10.0 * np.log10(np.maximum(decay, EPSILON))


def extrapolated_rt60(
    decay_db: np.ndarray,
    sample_rate: int,
    upper_db: float,
    lower_db: float,
) -> float:
    """Fit a decay interval and extrapolate its slope to 60 dB."""
    time_s = np.arange(len(decay_db), dtype=np.float64) / sample_rate
    mask = (decay_db <= upper_db) & (decay_db >= lower_db)
    if int(np.count_nonzero(mask)) < 20:
        return float("nan")

    slope, _ = np.polyfit(time_s[mask], decay_db[mask], 1)
    if slope >= 0:
        return float("nan")
    return float(-60.0 / slope)


def calculate_rir_metrics(rir: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Return EDT, T20-derived RT60, C50, and D50 for one aligned RIR."""
    decay_db = energy_decay_curve_db(rir)
    edt_s = extrapolated_rt60(decay_db, sample_rate, 0.0, -10.0)
    t20_rt60_s = extrapolated_rt60(decay_db, sample_rate, -5.0, -25.0)

    split = min(len(rir), int(round(0.050 * sample_rate)))
    squared = np.square(rir.astype(np.float64))
    early_energy = float(np.sum(squared[:split]))
    late_energy = float(np.sum(squared[split:]))
    total_energy = early_energy + late_energy

    c50_db = 10.0 * np.log10((early_energy + EPSILON) / (late_energy + EPSILON))
    d50_percent = 100.0 * early_energy / max(total_energy, EPSILON)
    return {
        "edt_s": float(edt_s),
        "t20_rt60_s": float(t20_rt60_s),
        "c50_db": float(c50_db),
        "d50_percent": float(d50_percent),
    }


def diagnose_reverberation(
    t20_rt60_s: float,
    c50_db: float,
) -> tuple[str, str, str]:
    """Return an explainable speech-oriented reverberation baseline."""
    rt60 = round(float(t20_rt60_s), 6)
    c50 = round(float(c50_db), 6)

    if rt60 <= 0.6 and c50 >= 3.0:
        return (
            "low_reverberation_risk",
            "Early sound dominates and the decay is short for this baseline.",
            "Investigate noise, HF loss, and alignment before treating reverberation.",
        )
    if rt60 <= 1.0 and c50 >= 0.0:
        return (
            "mild_reverberation_risk",
            "Reverberation may affect distant listeners but is not yet dominant.",
            "Verify occupied-room decay and loudspeaker directivity at audience positions.",
        )
    if rt60 <= 1.8 and c50 >= -3.0:
        return (
            "moderate_reverberation_risk",
            "Late energy is a plausible contributor to reduced consonant clarity.",
            "Improve direct-to-reverberant ratio and consider targeted absorption.",
        )
    return (
        "severe_reverberation_risk",
        "The decay or late-energy balance is likely to mask successive speech sounds.",
        "Prioritize acoustic treatment, loudspeaker pattern control, and closer coverage.",
    )


def apply_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve clean speech with an RIR and prevent output clipping."""
    reverberant = fftconvolve(clean, rir, mode="full")
    peak = float(np.max(np.abs(reverberant)))
    if peak > 0.99:
        reverberant = reverberant * (0.99 / peak)
    return reverberant.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled SpeechEval-LLM reverberation experiments."
    )
    parser.add_argument("--input", default="data/test.wav", help="Clean speech WAV")
    parser.add_argument(
        "--rt60s",
        nargs="+",
        type=float,
        default=[0.3, 0.8, 1.5, 2.5],
        help="Synthetic target RT60 values in seconds",
    )
    parser.add_argument("--seed", type=int, default=84, help="RIR random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_audio_dir = Path("data/generated")
    results_dir = Path("results")
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    clean, sample_rate = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    rows: list[dict[str, object]] = []
    decay_curves: list[tuple[str, np.ndarray]] = []

    for index, target_rt60_s in enumerate(args.rt60s):
        rng = np.random.default_rng(args.seed + index)
        rir = generate_synthetic_rir(sample_rate, target_rt60_s, rng)
        metrics = calculate_rir_metrics(rir, sample_rate)
        label, interpretation, intervention = diagnose_reverberation(
            metrics["t20_rt60_s"],
            metrics["c50_db"],
        )
        reverberant = apply_rir(clean, rir)

        rt_name = f"{target_rt60_s:g}".replace(".", "p")
        sf.write(output_audio_dir / f"rir_rt60_{rt_name}s.wav", rir, sample_rate)
        sf.write(
            output_audio_dir / f"test_reverb_rt60_{rt_name}s.wav",
            reverberant,
            sample_rate,
        )

        row = {
            "experiment": "Synthetic_Reverberation",
            "target_rt60_s": round(float(target_rt60_s), 3),
            "measured_edt_s": round(metrics["edt_s"], 3),
            "measured_t20_rt60_s": round(metrics["t20_rt60_s"], 3),
            "c50_db": round(metrics["c50_db"], 2),
            "d50_percent": round(metrics["d50_percent"], 1),
            "diagnosis": label,
            "interpretation": interpretation,
            "recommended_verification_or_action": intervention,
        }
        rows.append(row)
        decay_curves.append((f"Target {target_rt60_s:g} s", energy_decay_curve_db(rir)))

    csv_path = results_dir / "experiment_03_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, decay_db in decay_curves:
        time_s = np.arange(len(decay_db)) / sample_rate
        axis.plot(time_s, decay_db, label=label, linewidth=1.4)
    axis.set_xlim(0, max(args.rt60s) * 1.15)
    axis.set_ylim(-65, 1)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Energy decay (dB)")
    axis.set_title("Experiment 03 - Synthetic RIR Energy Decay Curves")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "experiment_03_decay_curves.png", dpi=170)
    plt.close(figure)

    print("====================================")
    print(" SpeechEval-LLM Reverb Experiment")
    print("====================================")
    for row in rows:
        print(
            f"Target {row['target_rt60_s']:>4} s | "
            f"T20 RT60 {row['measured_t20_rt60_s']:>5} s | "
            f"C50 {row['c50_db']:>6} dB | "
            f"{row['diagnosis']}"
        )
    print(f"\nSaved table: {csv_path}")
    print("Saved plot: results/experiment_03_decay_curves.png")


if __name__ == "__main__":
    main()
