"""
FourCastNet inference using earth2studio.

Install:
    pip install earth2studio[fcn]

Run:
    python fourcastnet_inference.py

Outputs saved to outputs/fcn_forecast.zarr and plots/ directory.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
import earth2studio.run as run

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INIT_TIME = "2025-03-10"   # forecast start date (YYYY-MM-DD)
NSTEPS = 20                 # number of 6-hourly steps  (20 = 5 days)
OUTPUT_ZARR = "outputs/fcn_forecast.zarr"

# Variables to plot  (must be in FCN output variables)
PLOT_VARS = {
    "u10m": ("10m U-wind",       "m/s",  "RdBu_r"),
    "v10m": ("10m V-wind",       "m/s",  "RdBu_r"),
    "t2m":  ("2m Temperature",   "K",    "RdYlBu_r"),
    "msl":  ("Mean Sea Level P", "Pa",   "viridis"),
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference():
    os.makedirs("outputs", exist_ok=True)

    print("Loading FourCastNet model (downloads weights on first run)...")
    package = FCN.load_default_package()
    model = FCN.load_model(package)

    print(f"Fetching GFS initial conditions for {INIT_TIME}...")
    data = GFS()

    io = ZarrBackend(OUTPUT_ZARR)

    print(f"Running inference: {NSTEPS} steps x 6h = {NSTEPS * 6}h forecast")
    io = run.deterministic([INIT_TIME], NSTEPS, model, data, io)
    print(f"Forecast saved to {OUTPUT_ZARR}")
    return io


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_forecast(zarr_path: str = OUTPUT_ZARR):
    os.makedirs("plots", exist_ok=True)

    ds = xr.open_zarr(zarr_path)
    print("Dataset variables:", list(ds.data_vars))
    print("Coordinates:", dict(ds.coords))

    times = ds.coords["time"].values if "time" in ds.coords else ds.coords["lead_time"].values

    for var, (title, units, cmap) in PLOT_VARS.items():
        if var not in ds:
            print(f"Skipping {var} — not found in output")
            continue

        da = ds[var]

        # pick the last time step for a snapshot
        da_final = da.isel(time=-1) if "time" in da.dims else da.isel(lead_time=-1)
        da_init  = da.isel(time=0)  if "time" in da.dims else da.isel(lead_time=0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        for ax, data_slice, label in zip(
            axes,
            [da_init, da_final],
            ["T+0h (initial)", f"T+{NSTEPS * 6}h (final)"],
        ):
            arr = data_slice.values.squeeze()
            im = ax.imshow(
                arr,
                origin="upper",
                cmap=cmap,
                aspect="auto",
            )
            ax.set_title(f"{title} — {label}")
            ax.set_xlabel("Longitude index")
            ax.set_ylabel("Latitude index")
            plt.colorbar(im, ax=ax, label=units, shrink=0.8)

        fig.suptitle(f"FourCastNet forecast initialised {INIT_TIME}", fontsize=13)
        plt.tight_layout()
        out_path = f"plots/fcn_{var}.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"Saved {out_path}")

    # Time series: global-mean of each variable
    fig, axes = plt.subplots(len(PLOT_VARS), 1, figsize=(10, 3 * len(PLOT_VARS)))
    if len(PLOT_VARS) == 1:
        axes = [axes]

    time_dim = "time" if "time" in ds.dims else "lead_time"

    for ax, (var, (title, units, _)) in zip(axes, PLOT_VARS.items()):
        if var not in ds:
            continue
        global_mean = ds[var].mean(dim=[d for d in ds[var].dims if d != time_dim])
        ax.plot(global_mean.values.squeeze())
        ax.set_title(f"Global mean {title}")
        ax.set_xlabel("Forecast step (x6h)")
        ax.set_ylabel(units)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/fcn_timeseries.png", dpi=120)
    plt.close()
    print("Saved plots/fcn_timeseries.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_inference()
    plot_forecast()
    print("\nDone. Check plots/ for output figures.")
