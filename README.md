# SpeechEval-LLM

LLM-assisted speech-intelligibility diagnosis prototype with Python audio analysis and controlled experiments.

## Project goal

SpeechEval-LLM investigates why speech can remain difficult to understand even when sound-pressure level is sufficient. The MVP is limited to five engineering causes:

- reverberation;
- late reflections;
- background noise;
- high-frequency loss;
- delay or polarity problems.

The system is designed as an evidence pipeline. DSP measurements and transparent baseline rules come first; a later LLM layer will explain structured evidence, rank likely causes, propose verification steps, and recommend interventions. The LLM is not asked to guess directly from raw audio.

## Diagnostic output target

Each completed diagnostic should report:

1. observation;
2. measured evidence;
3. likely cause and alternatives;
4. verification method;
5. prioritized intervention;
6. confidence and limitations.

## Current pipeline

```text
Speech WAV
   ↓
Python preprocessing
   ↓
Controlled degradation
   ↓
Feature extraction
   ↓
Rule-based baseline
   ↓
Experiment logging and visualization
   ↓
Structured LLM explanation (planned)
```

## Experiment 01 — High-Frequency Loss Detection

`main.py` applies a sixth-order 4 kHz low-pass filter to the same speech recording. It compares the original and processed signals using spectral centroid and spectrograms.

Observed first-test result:

- original spectral centroid: about 6730 Hz;
- HF-loss spectral centroid: about 1779 Hz;
- relative centroid drop: about 73.6%;
- HF-retention ratio: about 0.264.

This experiment establishes a same-source controlled comparison. Spectral centroid alone is not an intelligibility metric and cannot reliably diagnose HF loss across unrelated speakers or recordings.

## Experiment 02 — Controlled Noise and SNR

`experiment_02_snr.py` adds deterministic white noise at target SNR values of 20, 10, 5, and 0 dB. Because both the clean reference and injected noise are known, the script can verify the measured SNR before assigning a transparent baseline label.

The experiment saves:

- degraded WAV files under `data/generated/`;
- measured features and diagnoses in `results/experiment_02_results.csv`;
- a spectrogram comparison in `results/experiment_02_spectrograms.png`.

The baseline labels indicate noise risk, not a final speech-intelligibility score. A real venue diagnosis still requires measured room noise, speech level, reverberation, frequency response, and alignment evidence.

## Experiment 03 — Synthetic Reverberation and Room Metrics

`experiment_03_reverb.py` creates deterministic synthetic room impulse responses with target RT60 values of 0.3, 0.8, 1.5, and 2.5 seconds. It convolves the clean speech with each RIR and calculates room-acoustic evidence from the known impulse response:

- EDT from the 0 to -10 dB decay interval;
- T20-derived RT60 from the -5 to -25 dB interval;
- C50 early-to-late energy ratio;
- D50 early-energy percentage.

The first implementation test produced T20-derived RT60 values of approximately 0.311, 0.805, 1.488, and 2.505 seconds. C50 decreased from approximately +10.18 dB to -4.68 dB as the synthetic decay increased.

The experiment saves reverberant speech, RIR WAV files, a CSV evidence table, and plotted Schroeder energy-decay curves. Metrics are calculated from the RIR rather than inferred from speech alone.

## Engineering limitations

- Current degradations are synthetic and do not yet represent a measured conference room.
- White noise is a controlled baseline, while real HVAC, audience, traffic, and equipment noise are spectrally different.
- Known-reference SNR is available in the experiment but may not be available in field recordings.
- Spectral centroid is supporting evidence rather than proof of intelligibility.
- Real measured RIR import, ETC peak analysis, delay, polarity, and repeated listener evaluation are not yet implemented.
- LLM integration must be tested against a deterministic rule baseline for hallucinations and repeatability.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Place a clean, licensed or self-recorded mono/stereo speech WAV at:

```text
data/test.wav
```

Run Experiment 01:

```bash
python main.py
```

Run Experiment 02:

```bash
python experiment_02_snr.py
```

Custom SNR values and seed:

```bash
python experiment_02_snr.py --snrs 25 15 10 5 0 --seed 7
```

Run Experiment 03:

```bash
python experiment_03_reverb.py
```

Custom reverberation targets:

```bash
python experiment_03_reverb.py --rt60s 0.4 0.7 1.2 2.0 --seed 84
```

## Roadmap

- import and analyze real measured room impulse responses;
- Experiment 04: delay and polarity test cases;
- add ETC-based direct and late-reflection evidence;
- combine features into one structured diagnostic record;
- add LLM prompting and API integration;
- compare rule-based and LLM diagnoses;
- add repeatability, hallucination, and failure-case tests;
- compare selected outputs with experienced listeners or engineers.
