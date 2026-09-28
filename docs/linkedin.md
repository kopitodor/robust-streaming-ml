# LinkedIn copy (draft)

Draft text for the project owner's LinkedIn profile. Every number is copied from [RESULTS.md](../RESULTS.md). If you edit it, keep the scope statements: closed set of 10 commands, one model and seed, one machine, and no production or real-time claim.

## Project description

**Robust streaming keyword spotting.** A reproducible 10-command keyword classifier on Speech Commands v0.02 (CC BY 4.0), built as a portfolio project in ML and algorithms engineering. It covers:
- a verified, SHA-256-pinned dataset manifest;
- a 61,052-parameter log-mel CNN;
- calibrated held-out evaluation (0.9639 test accuracy on the 4,074-clip official test split);
- a deterministic suite of 15 corruption conditions;
- streaming-window inference;
- an ONNX export verified against PyTorch;
- a CPU latency benchmark whose protocol was committed before measuring;
- local demos, including a live-microphone mode that runs entirely on the user's machine.

Every reported number traces to a recorded run and a protocol committed before that run. Scope: closed-set command classification, not general speech recognition or production software. Stack: Python, PyTorch, ONNX, ONNX Runtime, NumPy, pytest, uv.

## Achievement bullets

- Achieved 0.9639 held-out test accuracy on a 10-command keyword task, measured by a pre-committed evaluation protocol on the 4,074-clip official Speech Commands v0.02 test split, by training a 61,052-parameter log-mel CNN from a SHA-256-pinned validated data manifest with deterministic seeding.
- Achieved reproducible robustness measurements for 15 synthetic corruption conditions (test accuracy from 0.9659 down to 0.4639), measured by per-condition degradation against the clean held-out baseline on all 4,074 test clips, by implementing seeded noise, reverberation, resampling, clipping and Opus-codec transforms behind a bit-exact reproduction gate.
- Achieved ONNX-to-PyTorch agreement within 1.8e-4 on logits with zero top-1 disagreements, measured by a predeclared 1e-2 tolerance check over 39,703 test and streaming windows in ONNX Runtime, by exporting the validated log-mel front end and CNN to ONNX opset 18 with fail-closed checkpoint and metadata checks.

## Post draft

I finished a portfolio project on reproducible streaming keyword spotting: a small log-mel CNN that classifies 10 spoken commands from Speech Commands v0.02.

What I focused on was making every number traceable. Each evaluation, robustness, export, and benchmark protocol was committed before its first run, and every run is tied to a clean git commit and SHA-256-pinned inputs.

A few measured results:
- 0.9639 accuracy on the 4,074-clip official test split;
- accuracy from 0.9659 down to 0.4639 across 15 synthetic corruption conditions (resampling through 2 kHz hurt most);
- the ONNX export matched PyTorch within 1.8e-4 on logits over 39,703 windows;
- 2.43 ms p50 per 1 s streaming window with ONNX Runtime on one laptop CPU thread.

One honest surprise: plain PyTorch was faster than my ONNX export on that laptop (p50 1.95 ms). A profile pointed at ONNX Runtime's STFT operator, at about 81% of kernel time. I reported it as measured rather than tuning it away after the fact.

Limits: a closed set of 10 commands, one model and seed, synthetic corruptions, and one machine. Not production software.

Code and write-up: [repository link]
