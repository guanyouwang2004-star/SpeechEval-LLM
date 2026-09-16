# SpeechEval-LLM

LLM-assisted speech intelligibility evaluation prototype with Python audio analysis and controlled experiments.

## Project goal

This project explores whether structured audio features can support reliable, explainable evaluation of speech intelligibility problems. The long-term goal is to combine Python-based audio preprocessing, acoustic feature extraction, rule-based baselines, LLM reasoning, and controlled experiments into a reproducible engineering prototype.

The current prototype focuses on a simple question:

> Can a controlled high-frequency loss be detected from measurable spectral changes before introducing an LLM layer?

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
Rule-based diagnosis
   ↓
Visualization and experiment logging
```

## Experiment 01 — High-Frequency Loss Detection

A 4 kHz low-pass filter is applied to the same speech recording to simulate high-frequency attenuation while keeping the source material constant.

The prototype then compares the original and processed signals using spectral centroid and spectrograms.

Observed result from the first controlled test:

- Original spectral centroid: about 6730 Hz
- HF-loss spectral centroid: about 1779 Hz
- Relative centroid drop: about 73.6%

The rule-based baseline reports a possible HF-loss condition when the relative centroid reduction is large.

## Why this matters

This is not intended to prove speech intelligibility from one feature. Spectral centroid is only one piece of evidence. The purpose of the first experiment is to establish a reproducible engineering workflow:

1. create a known audio degradation;
2. extract measurable features;
3. compare the result against a reference;
4. test a baseline diagnostic rule;
5. document limitations before adding an LLM.

## Engineering limitations

- Spectral centroid alone cannot determine speech intelligibility.
- A low centroid does not automatically imply high-frequency loss across different speakers or recordings.
- The current rule compares a degraded signal against a reference from the same source material.
- The present experiment is synthetic and does not yet represent a real conference-room measurement.
- Future versions should include SNR, reverberation, C50, RT60/EDT, delay/alignment features, and repeated evaluation.

## Next steps

- Experiment 02: controlled background noise / SNR degradation
- Experiment 03: reverberation-related degradation
- Combine multiple features into a unified diagnostic function
- Add structured LLM prompting and API integration
- Compare rule-based and LLM-based diagnoses
- Add repeatability, hallucination, and failure-case testing

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Place a speech WAV file at:

```text
data/test.wav
```

Then run:

```bash
python main.py
```

The script will create a processed HF-loss version and display spectrograms for comparison.
