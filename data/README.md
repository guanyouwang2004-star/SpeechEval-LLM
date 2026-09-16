# Data

Place local test audio here as `test.wav` when running the prototype.

Raw and processed audio files are intentionally not committed by default. This keeps the repository lightweight and avoids accidentally publishing copyrighted, private, or identifying recordings.

Current local workflow:

```text
data/test.wav
    ↓
main.py
    ↓
data/test_hf_loss.wav
```

For reproducible public experiments, future versions should use clearly licensed or self-recorded test material and document its source.
