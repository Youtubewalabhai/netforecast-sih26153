# NetForecast Evaluation Results

> **Data Source**: CIC-IDS-2018 Cleaned Telemetry  
> **Evaluation Split**: Strict Time-Based Partition (1483 test sequences)  
> **Validation Threshold Tuning**: World Model = 0.0500, Baseline = 0.0800  
> **Forecast Horizon**: K = 5 windows (50s forward)

## Step t+1 Forecasting Performance

| Model | Decision Threshold | F1-Score | Precision | Recall | False Positive Rate (FPR) | ROC-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **World Model (Ours)** | **0.05** | **0.8062** | **0.9718** | **0.6889** | **0.0087** | **0.9946** | **0.8995** |
| **Baseline (Logistic Reg.)** | 0.08 | 0.9727 | 0.9550 | 0.9911 | 0.0203 | 0.9985 | 0.9831 |

## Autoregressive Rollout Degradation (K-Step Ahead)

| Forecast Horizon | F1-Score | Precision | Recall | FPR | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **t+1 (10s)** | 0.8062 | 0.9718 | 0.6889 | 0.0087 | 0.9946 |
| **t+2 (20s)** | 0.7937 | 0.9711 | 0.6711 | 0.0087 | 0.9927 |
| **t+3 (30s)** | 0.7634 | 0.9660 | 0.6311 | 0.0097 | 0.9899 |
| **t+4 (40s)** | 0.7268 | 0.9668 | 0.5822 | 0.0087 | 0.9877 |
| **t+5 (50s)** | 0.6820 | 0.9673 | 0.5267 | 0.0077 | 0.9861 |

## Zero-Shot Attack Generalization Test (Holdout: Infilteration)
- **Sample Composition**: 424 attack windows, 0 benign windows (Total: 440)
- **FPR Explanation**: FPR is 0.0000 because the held-out attack slice comprises exclusively attack windows (0 benign samples, TN=0, FP=0), making false positive count strictly 0.
- **World Model Performance on Unseen Attack**: F1 = **0.7674** (Precision = 1.0000, Recall = 0.6226, FPR = 0.0000)
