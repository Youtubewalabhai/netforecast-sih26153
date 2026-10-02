#!/bin/bash
# NetForecast: CIC-IDS-2018 Data Download Script
# Team CHECK_MATE | SIH26153

set -e

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "======================================================================"
echo "Downloading CIC-IDS-2018 Selected Subset from AWS Open Data Bucket..."
echo "======================================================================"

# Check if AWS CLI is installed
if command -v aws &> /dev/null; then
    echo "AWS CLI detected. Syncing selected CSV files..."
    
    # Download 3 target files for laptop training
    aws s3 cp --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" "$RAW_DIR/" || true
    aws s3 cp --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv" "$RAW_DIR/" || true
    aws s3 cp --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv" "$RAW_DIR/" || true
    
    echo "Sync completed. Check files in $RAW_DIR:"
    ls -lh "$RAW_DIR"
else
    echo "AWS CLI is not installed."
    echo ""
    echo "MANUAL DOWNLOAD INSTRUCTIONS:"
    echo "1. Visit the official UNB CIC-IDS-2018 dataset repository:"
    echo "   https://www.unb.ca/cic/datasets/ids-2018.html"
    echo "   or direct AWS Open Data mirror."
    echo "2. Download the following 3 CSV files:"
    echo "   - Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv (FTP-BruteForce / SSH-Bruteforce)"
    echo "   - Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv (DoS-GoldenEye / DoS-Slowloris)"
    echo "   - Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv (Infiltration)"
    echo "3. Place them directly inside: netforecast/$RAW_DIR/"
    echo "======================================================================"
fi
