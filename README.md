# Federated Learning with Robust Aggregation & Attack Detection

A PyTorch implementation of a federated learning system that trains a shared global model across 10 decentralized clients — without centralizing any raw data — while actively detecting and isolating malicious client updates to maintain model integrity.

---

## Overview

Federated Learning (FL) enables collaborative model training across distributed nodes while preserving data privacy. This project simulates a realistic FL environment with:

- **10 independent clients**, each holding a private local dataset
- A **central server** that coordinates training rounds and aggregates client updates
- A **malicious client injection** mechanism (introduced at round 5) that submits poisoned model updates
- A **distance-based anomaly detection** layer on the server that identifies and filters out malicious updates before aggregation

The system achieves **98.9%+ global model accuracy** on MNIST under adversarial participation.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  SERVER                      │
│  - Maintains global model state              │
│  - Aggregates client updates (FedAvg)        │
│  - Detects & filters malicious updates       │
│  - Evaluates global model each round         │
└──────────┬──────────────────────────────────┘
           │  broadcast global model state
           ▼
┌──────────────────────────────────────────────┐
│   CLIENT 1 ... CLIENT 10 (decentralized)     │
│  - Each holds private local data partition   │
│  - Trains local CNN on global model weights  │
│  - Returns model update (state dict) only    │
│  - Raw data never leaves the client          │
└──────────────────────────────────────────────┘
```

### Attack Simulation
Starting at **Round 5**, Client 4 is replaced with a malicious actor that submits random noise as its model update — simulating a Byzantine attack. The server's anomaly detection layer computes the L2 distance of each client's update from the mean, and filters out updates exceeding 1.5× the average distance threshold before aggregating.

---

## Model Architecture — EnhancedCNN

A 3-layer convolutional neural network with batch normalization and dropout regularization:

| Layer | Type | Output |
|---|---|---|
| Conv1 | Conv2d(1→32, k=5) + BN + ReLU + MaxPool | 32×14×14 |
| Conv2 | Conv2d(32→64, k=3) + BN + ReLU + MaxPool | 64×7×7 |
| Conv3 | Conv2d(64→128, k=3) + BN + ReLU | 128×7×7 |
| FC1 | Linear(→256) + BN + ReLU + Dropout | 256 |
| FC2 | Linear(256→10) | 10 classes |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Clients | 10 (IID data split) |
| Rounds | 10 |
| Local Epochs | 1 per round |
| Batch Size | 32 |
| Optimizer | SGD (lr=0.01) |
| Loss | CrossEntropyLoss |
| Attack Introduced | Round 5 (Client 4, Byzantine noise) |

---

## Results

| Metric | Value |
|---|---|
| Global Model Accuracy (clean rounds 1–4) | ~98–99% |
| Global Model Accuracy (under attack, rounds 5–10) | **98.9%+** |
| Malicious client successfully isolated | ✅ |

The anomaly detection threshold filters out the poisoned update in every round it appears, demonstrating the system's robustness against Byzantine participants.

---

## Project Structure

```
├── Code.ipynb              # Full implementation notebook
│   ├── Cell 0              # Baseline FedAvg (FederatedAveraging, no attack)
│   └── Cell 1              # Full system: Client/Server classes + attack detection
├── Level3_model.pth        # Saved global model weights (post-training)
└── README.md
```

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision matplotlib
```

### Run
Open `Code.ipynb` in Jupyter and run all cells. The notebook will:
1. Download MNIST automatically
2. Partition data across 10 clients
3. Run 10 federated training rounds
4. Inject a malicious update at round 5
5. Output per-round accuracy and plot the accuracy curve

---

## Key Concepts Demonstrated

- **Federated Averaging (FedAvg)** — aggregating local model updates into a global model without sharing raw data
- **Byzantine Fault Tolerance** — detecting and isolating malicious/corrupted client updates using distance-based anomaly detection
- **Data Privacy** — only model weight updates (state dicts) are communicated; no raw data ever leaves a client
- **Distributed Systems Simulation** — Client and Server classes simulate realistic node coordination in a decentralized environment
- **CNN Design** — batch normalization, dropout regularization, and dynamic input shape inference

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `MNIST / CIFAR-10` · `Jupyter Notebook`
