<div align="center">

# DiskAnomaly

**An experimental pipeline for learning and visualizing anomalies in block I/O traces**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Trace Data](https://img.shields.io/badge/Data-blktrace%20%2B%20FIO-4C8CBF?style=flat-square)

</div>

## Overview

Storage devices are largely opaque from the host's point of view. Conventional trace viewers expose individual I/O events, but they do not make unusual behavior easy to identify at a glance.

DiskAnomaly explores a learning-based approach to this problem. It collects block I/O traces, converts them into fixed-length feature sequences, reconstructs those sequences with an LSTM-based variational model, and exposes the resulting anomaly values through a small Flask dashboard.

The project uses four features from each trace event:

| Feature | Meaning |
| --- | --- |
| `Timestamp` | Time associated with the I/O event |
| `IO_Type` | Encoded block-layer operation type |
| `Sector` | Target sector of the request |
| `Size` | Request size |

## Implementation

```text
FIO workloads
    │
    ▼
blktrace / blkparse
    │  Timestamp, I/O Type, Sector, Size
    ▼
Preprocessing
    ├── categorical I/O type encoding
    ├── numerical feature scaling
    └── fixed-length sequence construction
    │
    ▼
LSTM variational reconstruction model
    ├── LSTM encoder
    ├── latent distribution (μ, log σ²)
    ├── reparameterized latent sample
    ├── LSTM decoder
    └── LSTM discriminator output
    │
    ▼
Inference pipeline and Flask visualization
```

### Trace collection

The `pipeline/` directory contains wrappers for running FIO workloads and collecting block traces. The included FIO profiles cover sequential, random, mixed, multithreaded, latency-oriented, throughput-oriented, and cache-disabled workloads.

Parsed events are reduced to the four fields used by the model and can be streamed through the inference pipeline.

### Preprocessing

`preprocessing.py` reads the trace data, encodes block I/O operation types, scales the numerical fields, and groups events into sequences. The saved scaler in `checkpoint/scaler.pkl` is reused by the inference path.

### Model

The model in `model/model.py` is composed of:

- an LSTM encoder that produces `mu` and `logvar` for a latent distribution;
- a reparameterization step that samples the latent vector;
- an LSTM decoder that reconstructs the input sequence; and
- an LSTM discriminator that emits a real-or-fake score for the reconstruction.

The configured experiment uses an input dimension of 4, a hidden dimension of 64, and a latent dimension of 32. Training uses reconstruction MSE plus KL divergence, with RMSprop and a learning-rate range test.

### Inference and visualization

The execution path loads the model checkpoint and scaler, preprocesses incoming trace events, and calculates a reconstruction-based output. `app_flask.py` serves a dashboard and exposes the collected data as JSON through `/data`.

## Repository structure

```text
DiskAnomaly/
├── pipeline/          # FIO workloads, blktrace collection, and parsing
├── model/             # LSTM encoder, decoder, discriminator, and experiments
├── execute/           # Streaming preprocessing and inference path
├── templates/         # Flask dashboard templates
├── checkpoint/        # Model checkpoint and fitted scaler
├── preprocessing.py   # Dataset parsing and sequence preparation
├── train.py           # Model training and learning-rate search
├── main.py            # Offline analysis and plotting
└── app_flask.py       # Visualization server
```

## Experiment notes

Training traces were generated with FIO and collected through the block tracing pipeline. The original experiment was trained on an NVIDIA RTX A6000 without a separately curated validation set.

The central limitation is the absence of independently labeled normal and anomalous traces. Because the training data itself was not divided by behavior class, the reconstruction output cannot yet be interpreted as a validated anomaly boundary. This repository should therefore be treated as an exploratory prototype rather than a production anomaly detector.

## Visual results

### Conventional trace output

![Raw trace view](https://github.com/user-attachments/assets/bf7787ea-69b0-4c9c-9dc9-104803a80344)

![Trace analysis view](https://github.com/user-attachments/assets/0be392a3-00fc-4af6-945a-8a4d86158250)

### Model architecture

![Model architecture](https://github.com/user-attachments/assets/2717c17b-449f-4801-b156-93c6c3c5adb7)

### Experimental output

![Experimental output](https://github.com/user-attachments/assets/e12336e3-962a-48c0-8e03-def975211e3a)

## Reference

Y. Wang, “A One-Class Anomaly Detection Method for Drives based on Adversarial Auto-Encoder,” *2022 IEEE 24th International Conference on High Performance Computing & Communications*.
