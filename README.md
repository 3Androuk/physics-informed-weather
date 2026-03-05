# Physics-Informed Weather Prediction Baselines

FNO vs AFNO baseline comparison on ERA5 geopotential at 500 hPa (Z500), using
[WeatherBench 2](https://sites.research.google/weatherbench/) data at 1.5-degree
resolution (120 x 240 grid).

## Data

ERA5 Z500 data is automatically downloaded from the
[WeatherBench 2](https://github.com/google-research/weatherbench2) Zarr store
on Google Cloud Storage (public, no authentication required). Data is cached
locally in `cache/` after the first run.

- Resolution: 1.5 degrees (121 lat x 240 lon)
- Temporal resolution: 6-hourly
- Training period: 1979-2015
- Validation period: 2016-2017
- Climatology: WB2 pre-computed 1990-2019 hourly climatology

## Setup

Requires WSL (Ubuntu) with Python 3.12 and a CUDA-capable GPU.

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Full training (40 epochs, evaluates at 24h/72h/120h lead times)
python train_baselines.py

# Quick test run
python train_baselines.py --epochs 2 --lead-hours 24

# Custom configuration
python train_baselines.py \
    --train-years 1979 2015 \
    --val-years 2016 2017 \
    --train-lead-hours 24 \
    --lead-hours 24 72 120 \
    --epochs 40 \
    --batch-size 32
```

## Project Structure

```
data.py              # WB2 data loading, caching, normalization
evaluate.py          # Evaluation metrics (lat-weighted RMSE, ACC)
train_baselines.py   # Main training script
requirements.txt     # Python dependencies
```

## Evaluation Metrics

- **Latitude-weighted RMSE** (m^2/s^2): Root mean squared error weighted by
  cosine of latitude to account for grid cell area differences.
- **ACC** (Anomaly Correlation Coefficient): Correlation between forecast
  anomalies and truth anomalies relative to climatology. ACC > 0.6 indicates
  useful synoptic-scale skill.

Both metrics follow WeatherBench 2 conventions (Rasp et al., 2024).

## References

- Rasp, S., Hoyer, S., et al. (2024). WeatherBench 2: A Benchmark for the Next
  Generation of Data-Driven Weather Models. *JAMES*.
  https://doi.org/10.1029/2023MS004019
- Li, Z., et al. (2021). Fourier Neural Operator for Parametric Partial
  Differential Equations. *ICLR*. https://arxiv.org/abs/2010.08895
- Guibas, J., et al. (2022). Adaptive Fourier Neural Operators: Efficient Token
  Mixers for Transformers. *ICLR*. https://arxiv.org/abs/2111.13587
