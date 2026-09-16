import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# =========================
# 1. Load original speech
# =========================
audio, sr = librosa.load(
    "data/test.wav",
    sr=48000,
    mono=True,
)

# =========================
# 2. Simulate high-frequency loss
#    with a 4 kHz low-pass filter
# =========================
cutoff = 4000

sos = butter(
    N=6,
    Wn=cutoff,
    btype="lowpass",
    fs=sr,
    output="sos",
)

audio_hf_loss = sosfilt(sos, audio)

# =========================
# 3. Save processed audio
# =========================
sf.write(
    "data/test_hf_loss.wav",
    audio_hf_loss,
    sr,
)

# =========================
# 4. Extract spectral centroid
# =========================
centroid_original = np.mean(
    librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
    )
)

centroid_hf_loss = np.mean(
    librosa.feature.spectral_centroid(
        y=audio_hf_loss,
        sr=sr,
    )
)

# =========================
# 5. Calculate relative change
# =========================
ratio = centroid_hf_loss / centroid_original
drop_percent = (1 - ratio) * 100

# =========================
# 6. Print analysis result
# =========================
print("====================================")
print(" SpeechEval-LLM HF Loss Experiment")
print("====================================")

print(
    "Original Spectral Centroid:",
    round(float(centroid_original), 2),
    "Hz",
)

print(
    "HF Loss Spectral Centroid:",
    round(float(centroid_hf_loss), 2),
    "Hz",
)

print(
    "Centroid Change:",
    round(float(centroid_hf_loss - centroid_original), 2),
    "Hz",
)

print(
    "HF Retention Ratio:",
    round(float(ratio), 3),
)

print(
    "HF Drop Percent:",
    round(float(drop_percent), 1),
    "%",
)

# =========================
# 7. Rule-based baseline diagnosis
# =========================
print("\n=== Diagnosis ===")

if ratio < 0.5:
    print("Possible HF Loss detected")
    print(
        "Evidence: Spectral centroid decreased by",
        round(float(drop_percent), 1),
        "%",
    )
    print(
        "Interpretation: High-frequency speech information "
        "may have been significantly reduced."
    )
elif ratio < 0.8:
    print("Moderate HF reduction detected")
    print(
        "Evidence: Spectral centroid decreased by",
        round(float(drop_percent), 1),
        "%",
    )
else:
    print("No obvious HF Loss detected")

# =========================
# 8. Plot original spectrogram
# =========================
D_original = librosa.stft(audio)
S_original_db = librosa.amplitude_to_db(
    np.abs(D_original),
    ref=np.max,
)

plt.figure(figsize=(10, 5))
librosa.display.specshow(
    S_original_db,
    sr=sr,
    x_axis="time",
    y_axis="log",
)
plt.title("Original Speech")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.show()

# =========================
# 9. Plot HF-loss spectrogram
# =========================
D_hf = librosa.stft(audio_hf_loss)
S_hf_db = librosa.amplitude_to_db(
    np.abs(D_hf),
    ref=np.max,
)

plt.figure(figsize=(10, 5))
librosa.display.specshow(
    S_hf_db,
    sr=sr,
    x_axis="time",
    y_axis="log",
)
plt.title("HF Loss Speech - 4 kHz Low-pass")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.show()
