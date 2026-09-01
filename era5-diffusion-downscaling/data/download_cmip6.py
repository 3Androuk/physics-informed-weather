"""Download CMIP6 daily fields matching the 20-channel config — deployment demo.

Purpose: real-GCM input for the deployment demonstration. Pulls one CMIP6
model's historical daily output for the SAME 20 variables as config
data.variables (surface: tas/uas/vas/psl/prw; pressure-level: zg/ta/ua/va/hus
at 500/700/850 hPa), regrids it onto the project's COARSE grid (the block-mean
grid of the ERA5 band at --ratio), and saves a (T, C, h, w) float32 array
ready to feed the sampler as the coarse observation. There is no fine truth:
evaluation against ERA5 is distributional (spectra, histograms, quantiles).

Data source: the Pangeo CMIP6 mirror on Google Cloud Storage (anonymous zarr,
same access path as the WeatherBench2 download). Default model
MPI-ESM1-2-HR (~1 deg = the project's 4x scale), r1i1p1f1, historical
2007-2014 (historical ends in 2014).

Unit notes (everything else already matches ERA5):
  * zg is geopotential HEIGHT in m; ERA5 geopotential is m2/s2 -> multiply
    by g = 9.80665.
  * CMIP6 `day` fields are DAILY MEANS; the model was trained on 6-hourly
    snapshots. That temporal-averaging shift is part of what the demo
    documents — pair it with a coarsened daily-mean ERA5 control.

Resumable: each channel is written into an on-disk memmap and marked with a
.done file; a rerun skips finished channels.

Run (BriCS login node, no credentials needed):
    python -m data.download_cmip6 --config config/wb2_20var.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import channel_labels, channel_specs, ensure_dir, load_config  # noqa: E402

CATALOG_CSV = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
G = 9.80665

# ERA5/WB2 variable name -> (CMIP6 variable_id, multiplicative factor to ERA5
# units). Pressure-level variables keep their config level (hPa -> Pa here).
CMIP6_MAP = {
    "2m_temperature": ("tas", 1.0),
    "10m_u_component_of_wind": ("uas", 1.0),
    "10m_v_component_of_wind": ("vas", 1.0),
    "mean_sea_level_pressure": ("psl", 1.0),
    "total_column_water_vapour": ("prw", 1.0),
    "geopotential": ("zg", G),          # height (m) -> geopotential (m2/s2)
    "temperature": ("ta", 1.0),
    "u_component_of_wind": ("ua", 1.0),
    "v_component_of_wind": ("va", 1.0),
    "specific_humidity": ("hus", 1.0),
}


def cmip6_spec(spec: dict) -> tuple:
    """Config channel spec -> (cmip6 variable_id, plev in Pa or None, factor)."""
    name = spec["name"]
    if name not in CMIP6_MAP:
        raise KeyError(f"no CMIP6 mapping for variable: {name}")
    var, factor = CMIP6_MAP[name]
    level = spec.get("level")
    return var, (None if level is None else float(level) * 100.0), factor


def coarse_grid(lat: np.ndarray, lon: np.ndarray, ratio: int) -> tuple:
    """Block-mean coarse grid of the fine ERA5 band grid at `ratio`.

    Trims trailing rows/cols so the counts divide evenly (matching what
    coarsen() does to the fields), then averages each block — the coarse
    cell-centre coordinates the sampler's coarse observation lives on.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    h = (len(lat) // ratio) * ratio
    w = (len(lon) // ratio) * ratio
    return (lat[:h].reshape(-1, ratio).mean(axis=1),
            lon[:w].reshape(-1, ratio).mean(axis=1))


def load_fine_grid(cfg: dict) -> tuple:
    """The ERA5 band grid this project runs on, from saved coords artifacts."""
    for base, fname in ((cfg["paths"].get("patch_dir"), "coords_full.npz"),
                        (cfg["paths"].get("raw_dir"), "coords.npz")):
        if base and (Path(base) / fname).exists():
            cf = np.load(Path(base) / fname)
            return cf["lat"], cf["lon"]
    raise FileNotFoundError(
        "no coords_full.npz / coords.npz found — run the ERA5 download or "
        "make_patches first so the target grid exists")


def _open_cmip6(zstore: str, timeout):
    import xarray as xr
    storage = {"token": "anon"}
    if timeout:
        storage["requests_timeout"] = timeout
    try:
        return xr.open_zarr(zstore, storage_options=storage, consolidated=True)
    except TypeError:
        storage.pop("requests_timeout", None)
        return xr.open_zarr(zstore, storage_options=storage, consolidated=True)


def _find_zstore(catalog, source, experiment, member, variable):
    rows = catalog[(catalog.source_id == source)
                   & (catalog.experiment_id == experiment)
                   & (catalog.member_id == member)
                   & (catalog.table_id == "day")
                   & (catalog.variable_id == variable)]
    if len(rows) == 0:
        raise LookupError(
            f"{source}/{experiment}/{member}/day/{variable} not in the Pangeo "
            f"catalog — try another --member (r2i1p1f1, ...) or --source")
    # Prefer the native grid; otherwise take the first listing.
    gn = rows[rows.grid_label == "gn"]
    return (gn.iloc[0] if len(gn) else rows.iloc[0]).zstore


def _regrid(da, lat_c, lon_c):
    """Bilinear interp onto the coarse grid, with longitude wraparound.

    CMIP6 lon is 0..360 ascending; pad one wrapped column so targets between
    lon[-1] and 360 interpolate instead of turning NaN at the seam.
    """
    import xarray as xr
    lon_name = "lon" if "lon" in da.dims else "longitude"
    lat_name = "lat" if "lat" in da.dims else "latitude"
    pad = da.isel({lon_name: 0})
    pad = pad.assign_coords({lon_name: float(da[lon_name][0]) + 360.0})
    da = xr.concat([da, pad], dim=lon_name)
    out = da.interp({lat_name: lat_c, lon_name: lon_c % 360.0},
                    method="linear")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--source", default="MPI-ESM1-2-HR",
                    help="CMIP6 source_id (~1 deg default = the 4x scale)")
    ap.add_argument("--experiment", default="historical")
    ap.add_argument("--member", default="r1i1p1f1")
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2014-12-31",
                    help="historical ends 2014-12-31")
    ap.add_argument("--ratio", type=int, default=4,
                    help="which coarse grid of the project to target "
                         "(4 -> 1 deg, the closest to ~1 deg CMIP6 models)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <raw_dir>/../cmip6_<source>)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--batch", type=int, default=180,
                    help="days per streamed read")
    args = ap.parse_args()

    import pandas as pd

    cfg = load_config(args.config)
    specs = channel_specs(cfg["data"])
    labels = channel_labels(cfg["data"])
    fine_lat, fine_lon = load_fine_grid(cfg)
    lat_c, lon_c = coarse_grid(fine_lat, fine_lon, args.ratio)
    out_dir = ensure_dir(args.out or (Path(cfg["paths"]["raw_dir"]).parent
                                      / f"cmip6_{args.source}"))
    print(f"target coarse grid: {len(lat_c)} x {len(lon_c)} "
          f"(ratio {args.ratio} of the {len(fine_lat)} x {len(fine_lon)} band)")

    print("reading Pangeo CMIP6 catalog ...")
    catalog = pd.read_csv(CATALOG_CSV)

    # Establish the time axis once, from the first channel's store.
    var0, _, _ = cmip6_spec(specs[0])
    ds0 = _open_cmip6(_find_zstore(catalog, args.source, args.experiment,
                                   args.member, var0), args.timeout)
    times = ds0.time.sel(time=slice(args.start, args.end)).values
    t_total = len(times)
    if t_total == 0:
        raise SystemExit(f"no timesteps in [{args.start}, {args.end}] — "
                         f"check --experiment/--start/--end")
    print(f"{args.source} {args.experiment} {args.member}: {t_total} days "
          f"{args.start}..{args.end}, {len(specs)} channels")

    shape = (t_total, len(specs), len(lat_c), len(lon_c))
    mm_path = out_dir / "cmip6_coarse.npy"
    if mm_path.exists():
        mm = np.lib.format.open_memmap(mm_path, mode="r+")
        assert mm.shape == shape, f"existing {mm_path} has shape {mm.shape}, expected {shape}"
    else:
        mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32,
                                       shape=shape)

    for c, spec in enumerate(specs):
        marker = out_dir / f".{labels[c]}.done"
        if marker.exists():
            print(f"  [{c + 1}/{len(specs)}] {labels[c]}: cached, skip")
            continue
        var, plev, factor = cmip6_spec(spec)
        ds = _open_cmip6(_find_zstore(catalog, args.source, args.experiment,
                                      args.member, var), args.timeout)
        da = ds[var].sel(time=slice(args.start, args.end))
        if plev is not None:
            da = da.sel(plev=plev, method="nearest", tolerance=100.0)
        print(f"  [{c + 1}/{len(specs)}] {labels[c]} <- {var}"
              f"{'' if plev is None else f'@{int(plev)}Pa'} x{factor:g}")
        for t0 in range(0, t_total, args.batch):
            t1 = min(t0 + args.batch, t_total)
            block = _regrid(da.isel(time=slice(t0, t1)), lat_c, lon_c)
            vals = np.asarray(block.values, dtype=np.float32) * factor
            if not np.isfinite(vals).all():
                raise SystemExit(f"non-finite values in {labels[c]} days "
                                 f"{t0}..{t1} — inspect the store")
            mm[t0:t1, c] = vals
            print(f"      days {t0:5d}..{t1:5d} / {t_total}", flush=True)
        mm.flush()
        marker.touch()

    np.savez(out_dir / "cmip6_meta.npz",
             lat=lat_c, lon=lon_c, ratio=np.array([args.ratio]),
             times=np.array([str(t)[:10] for t in times]),
             channels=np.array(labels))
    with open(out_dir / "cmip6_meta.json", "w") as f:
        json.dump({"source": args.source, "experiment": args.experiment,
                   "member": args.member, "start": args.start,
                   "end": args.end, "ratio": args.ratio,
                   "shape": list(shape), "channels": labels,
                   "notes": "daily means; zg converted to m2/s2; coarse grid "
                            "= block means of the ERA5 band grid"}, f, indent=2)
    print(f"done -> {mm_path} {shape} + cmip6_meta.npz/json")
    print("next: quantile-map against coarsened daily-mean ERA5 on the "
          "overlap, then feed as the coarse observation to the sampler")


if __name__ == "__main__":
    main()
