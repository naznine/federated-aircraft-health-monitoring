# Federated Learning for Aircraft Engine Health Monitoring Using C-MAPSS

---

## Overview

This repository implements a federated learning pipeline for **Prognostics and Health Management (PHM)** of aircraft turbofan engines. The core problem: multiple airlines cannot share sensitive engine degradation data, yet all benefit from collaborative learning. Federated Learning (FL) solves this by training models locally and sharing only weights — never raw data.

The pipeline addresses:
- **Dual-task prediction**: simultaneous RUL regression and early fault classification
- **Federated learning**: 4 airline clients (one per C-MAPSS subset) collaborating via FedAvg
- **Three research questions**: class imbalance (RQ2), concept drift (RQ4), and non-IID validation bias (RQ5)

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**  
Source: https://ieee-dataport.org/documents/c-mapss-dataset

| Subset | Engines (Train) | Operating Conditions | Fault Modes | FL Client |
|--------|----------------|---------------------|-------------|-----------|
| FD001  | 100            | 1                   | 1           | Client 1  |
| FD002  | 260            | 6                   | 1           | Client 2  |
| FD003  | 100            | 1                   | 2           | Client 3  |
| FD004  | 249            | 6                   | 2           | Client 4  |

**Key parameters:**
- `WINDOW_SIZE = 30` cycles
- `RUL_CAP = 125` cycles
- `FAULT_THRESHOLD = 30` cycles
- Dropped sensors: `s1, s5, s10, s16, s18, s19` (near-zero variance)

---

## Repository Structure

```
├── Try-1-visualization_analysis.ipynb          # EDA and statistical analysis
├── Try-2-1D-CNN.ipynb                          # Centralized 1D-CNN baseline (FD001)
├── Try-3-1D-CNN-BiLSTM-MultiHeadAttention-     # Full architecture, all 4 FDs
│   DualOutputHeads.ipynb
├── Try-4-FLwith1D-CNN-BiLSTM-                  # FedAvg federated baseline
│   MultiHeadAttention-DualOutputHeads.ipynb
├── Try-5-RQ2.ipynb                             # RQ2: Fault-aware aggregation
├── Try-6-RQ4.ipynb                             # RQ4: Concept drift detection
├── Try-7-RQ5.ipynb                             # RQ5: Non-IID validation bias
└── README.md
```

---

## Notebooks

### Try-1 — EDA and Statistical Analysis
- Descriptive statistics across all 4 FD subsets
- Engine lifecycle distribution
- Sensor variance analysis (justifies sensor dropping)
- Sensor-RUL correlation analysis (justifies feature selection)
- Operating condition clustering (FD002/FD004)
- Cross-client fault ratio comparison (motivates FL and RQ2)

### Try-2 — 1D-CNN Centralized Baseline
- Initial baseline on FD001 only
- Dual-head architecture: RUL regression + fault classification
- Establishes starting point for architecture iteration

### Try-3 — 1D-CNN + BiLSTM + MultiHeadAttention (Centralized)
**Architecture:** `Input → Conv1D(×3) → BiLSTM → MultiHeadAttention → GlobalAvgPool → Dense → [RUL output | Fault output]`

- Cluster-based normalization for multi-condition FDs (FD002, FD004)
- Weighted BCE loss with sqrt-dampened pos_weight for class imbalance
- Evaluated on all 4 FD subsets independently

**Centralized results (last window per engine):**

| Dataset | RUL MAE | R²    | Recall | F1     | AUPRC  |
|---------|---------|-------|--------|--------|--------|
| FD001   | 10.98   | 0.850 | 0.920  | 0.885  | 0.977  |
| FD002   | 14.70   | 0.814 | 1.000  | 0.891  | 0.982  |
| FD003   | 14.16   | 0.739 | 1.000  | 0.851  | 0.959  |
| FD004   | 16.81   | 0.765 | 1.000  | 0.848  | 0.956  |

### Try-4 — FedAvg Federated Baseline
- 4 FL clients (one per FD subset)
- FedAvg aggregation weighted by client dataset size
- `FL_ROUNDS = 25`, `LOCAL_EPOCHS = 8`
- Per-client evaluation after each round

**FedAvg baseline results (last window per engine):**

| Dataset | RUL MAE | R²    | Recall | F1     | AUPRC  |
|---------|---------|-------|--------|--------|--------|
| FD001   | 13.54   | 0.801 | 0.880  | 0.898  | 0.981  |
| FD002   | 14.05   | 0.810 | 0.869  | 0.898  | 0.970  |
| FD003   | 15.69   | 0.706 | 0.900  | 0.857  | 0.960  |
| FD004   | 14.61   | 0.792 | 0.906  | 0.857  | 0.938  |

### Try-5 — RQ2: Class Imbalance Across Clients
**Research question:** How to design aggregation methods that protect weak fault signals from data-scarce clients?

**Approach:** Replace size-based FedAvg weights with fault-aware weights:
```
fault_weight(client) = 1 / (fault_ratio + ε)   # inverse: rarer faults → higher weight
final_weight = 0.5 × size_weight + 0.5 × fault_weight
```

**Key findings (last window per engine):**

| Dataset | Baseline Recall | RQ2 Recall | Change | Baseline F1 | RQ2 F1 | Change |
|---------|----------------|------------|--------|-------------|--------|--------|
| FD001   | 0.880          | 0.880      | —      | 0.898       | 0.898  | —      |
| FD002   | 0.869          | 0.869      | —      | 0.898       | 0.876  | ↓0.022 |
| FD003   | 0.900          | **1.000**  | ↑0.100 | 0.857       | **0.909** | ↑0.052 |
| FD004   | 0.906          | **0.943**  | ↑0.038 | 0.857       | 0.833  | ↓0.024 |

Fault-aware aggregation improves recall for the most fault-scarce clients (FD003/FD004) — a favorable safety tradeoff where missed faults are more costly than false alarms.

### Try-6 — RQ4: Concept Drift Detection
**Research question:** If a fault mode shift is injected mid-training, how quickly does AUPRC degrade and can a drift detector recover performance within 10 FL rounds?

**Approach:**
- Inject drift on FD001 at round 13 (swap training data to FD002)
- Monitor per-client AUPRC every round
- Trigger higher learning rate (3×) on drift detection
- Track recovery within 10 rounds

**Findings:**
- Drift successfully detected on FD001 at round 17
- AUPRC dropped from 0.973 (pre-drift) to 0.933 (minimum)
- Recovery gain of 0.025 AUPRC observed within 10 rounds
- **Limitation:** threshold sensitivity (0.03) caused false alarms on non-drift clients — adaptive thresholding is recommended as future work

### Try-7 — RQ5: Non-IID Validation Bias
**Research question:** How much does distributional mismatch distort aggregation weights, and how does accounting for it change per-operator RUL accuracy?

**Approach:**
1. Build 4×4 cross-evaluation matrix (each client's model scored on every other client's data)
2. Compute distributional distance via Maximum Mean Discrepancy (MMD)
3. Correct aggregation weights by discounting cross-scores proportional to MMD distance

**Key finding:** Models trained on simple single-condition data (FD001/FD003) score poorly on complex multi-condition data (FD002/FD004) — not because the models are poor, but due to distributional mismatch. Distance-corrected weights reduce this systematic bias.

---

## Model Architecture

```
Input (30 cycles × 18 sensors)
        ↓
Conv1D(64, k=5) → BN → MaxPool → Dropout(0.2)
Conv1D(128, k=3) → BN → MaxPool → Dropout(0.2)
Conv1D(128, k=3) → BN
        ↓
Bidirectional LSTM (64 units)
        ↓
Multi-Head Attention (4 heads, key_dim=16)
        ↓
GlobalAveragePooling1D
        ↓
Dense(64, ReLU) → Dropout(0.2)
        ↓
[RUL output: Dense(1, linear)]   [Fault output: Dense(1, sigmoid)]
```

**Loss:** MSE for RUL + Weighted BCE for fault (sqrt-dampened pos_weight)  
**Optimizer:** Adam (lr=1e-3)  
**Loss weights:** RUL=1.0, Fault=1.5

---

## Environment

**Google Colab (training):**
- Python 3.12
- TensorFlow 2.18 / Keras 3.x
- T4 GPU

**Local (VSCode):**
- Python 3.x
- TensorFlow 2.x / Keras 2.x

**Key packages:** `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Results Summary

| Experiment | Best RUL MAE | Best Recall | Best AUPRC |
|-----------|-------------|-------------|------------|
| Centralized (FD001) | 10.98 | 0.920 | 0.977 |
| FedAvg baseline | 13.54 | 0.906 | 0.981 |
| FedAvg + RQ2 | 13.47 | **1.000** (FD003) | 0.981 |
| FedAvg + RQ4 | 14.93 | 0.920 | 0.970 |
| FedAvg + RQ5 | cross-eval analysis | — | — |

---

## Research Questions Addressed

| RQ | Title | Status |
|----|-------|--------|
| RQ2 | Class Imbalance across Clients |
| RQ4 | Concept Drift Detection |
| RQ5 | Non-IID Validation Bias |

---

