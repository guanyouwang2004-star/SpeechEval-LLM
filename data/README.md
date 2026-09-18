# Data

Place local test audio here as `test.wav` when running the prototype.

Raw and processed audio files are intentionally excluded from Git. This keeps the repository lightweight and avoids accidentally publishing copyrighted, private, or identifying recordings.

Current local workflows:

```text
data/test.wav
    ├── main.py
    │      └── data/test_hf_loss.wav
    └── experiment_02_snr.py
           ├── data/generated/test_snr_20db.wav
           ├── data/generated/test_snr_10db.wav
           ├── data/generated/test_snr_5db.wav
           └── data/generated/test_snr_0db.wav

data/test.wav
    └── experiment_03_reverb.py
           ├── data/generated/rir_rt60_0p3s.wav
           ├── data/generated/rir_rt60_0p8s.wav
           ├── data/generated/rir_rt60_1p5s.wav
           ├── data/generated/rir_rt60_2p5s.wav
           └── data/generated/test_reverb_rt60_*.wav
```

For reproducible public experiments, use clearly licensed or self-recorded speech and document the speaker, language, recording chain, sampling rate, and source license without exposing personal information.
