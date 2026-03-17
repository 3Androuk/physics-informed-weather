# Physics-Informed Weather Prediction Baselines

FNO vs AFNO baseline comparison on ERA5 Z500 and T850, using
[WeatherBench 2](https://sites.research.google/weatherbench/) data at 1.5-degree
resolution (120 x 240 grid).

## Data

ERA5 data (Z500 + T850) comes from the
[WeatherBench 2](https://github.com/google-research/weatherbench2) project,
hosted on Google Cloud Storage (public, no account required).

**Automatic (recommended):** Just run the training script. It streams the
required variables from GCS on first run and caches them locally in `cache/`
(~6 GB for the full 1979–2015 training set). No manual steps needed.

**Manual download:** The data is stored as Zarr archives (directories of
chunked files, not a single downloadable file). To download manually, use
the Python snippet below:

```python
import xarray as xr
import numpy as np

ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
ds = xr.open_zarr(ERA5_ZARR, storage_options=dict(token="anon"))

# Download Z500 and T850 for desired years
z500 = ds["geopotential"].sel(level=500, time=slice("1979", "2015")).compute()
t850 = ds["temperature"].sel(level=850, time=slice("1979", "2015")).compute()

# Save locally
z500.to_netcdf("z500_1979_2015.nc")
t850.to_netcdf("t850_1979_2015.nc")
```

**Links:**
- [WB2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) — full list of datasets and access instructions
- [ERA5 dataset on GCS](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5) — browse the reanalysis data
- [Climatology on GCS](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5-hourly-climatology) — used for ACC evaluation
- [WB2 GitHub](https://github.com/google-research/weatherbench2) — benchmark code and documentation

**Specs:**
- Resolution: 1.5 degrees (121 lat x 240 lon)
- Variables: geopotential @ 500 hPa (Z500), temperature @ 850 hPa (T850)
- Temporal resolution: 6-hourly
- Training period: 1979-2015
- Validation period: 2016-2017

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

- **Latitude-weighted RMSE**: Root mean squared error weighted by cosine of
  latitude to account for grid cell area differences. Units: m^2/s^2 for Z500, K for T850.
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
