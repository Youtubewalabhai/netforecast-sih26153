# NetForecast Evaluation Results

> **Data Source**: CIC-IDS-2018 Cleaned Telemetry  
> **Evaluation Split**: Strict Time-Based Partition (1483 test sequences)  
> **Forecast Horizon**: K = 5 windows (50s forward)

## Step t+1 Forecasting Performance

| Model | F1-Score | Precision | Recall | False Positive Rate (FPR) | ROC-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **World Model (Ours)** | **0.6667** | **0.9827** | **0.5044** | **0.0039** | **0.9946** | **0.8469** |
| **Baseline (Logistic Reg.)** | 0.2524 | 1.0000 | 0.1444 | 0.0000 | 0.9985 | 0.7404 |

## Autoregressive Rollout Degradation (K-Step Ahead)

| Forecast Horizon | F1-Score | Precision | Recall | FPR | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **t+1 (10s)** | 0.6667 | 0.9827 | 0.5044 | 0.0039 | 0.9946 |
| **t+2 (20s)** | 0.6488 | 0.9820 | 0.4844 | 0.0039 | 0.9927 |
| **t+3 (30s)** | 0.6037 | 0.9612 | 0.4400 | 0.0077 | 0.9899 |
| **t+4 (40s)** | 0.5665 | 0.9577 | 0.4022 | 0.0077 | 0.9877 |
| **t+5 (50s)** | 0.5234 | 0.9586 | 0.3600 | 0.0068 | 0.9861 |

## Zero-Shot Attack Generalization Test (Holdout: Infilteration)
- **Status**: Evaluated on 440 holdout windows.
- **World Model F1 on Unseen Attack**: 0.6098 (Recall: 0.4387, FPR: 0.0000)
