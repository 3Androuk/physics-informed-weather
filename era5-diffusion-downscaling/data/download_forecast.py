"""Download coarse forecast fields + matching ERA5 truth — deployment demo.

Motivation: downscaling coarse-run / coarse-disseminated forecast products.
Unlike CMIP6, forecasts are time-synchronized with reality, so PAIRED truth
exists and the full pointwise metric suite applies.

Sources (WeatherBench2 GCS, anonymous — verified listings):
  * hres          gs://weatherbench2/datasets/hres/
                  2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr
                  IFS HRES regridded to 1.5 deg (the dissemination-grade coarse
                  product of a 9 km run). Has 19 of the 20 config channels —
                  everything except total_column_water_vapour (masked as
                  unobserved). Inits 00/12 UTC 2016-2022: choosing 2016-2017
                  makes the truth the project's existing TEST years.
  * neuralgcm_ens gs://weatherbench2/datasets/neuralgcm_ens/
                  2020-240x121_equiangular_with_poles_conservative.zarr
                  NeuralGCM-ENS, natively RUN at 1.4 deg (true coarse-run
                  system, the compute-savings framing), 50 members, 2020.
                  Pressure-level channels only: 15 of 20 observed, the 5
                  surface channels masked (channel inpainting).

1.5 deg / 0.25 deg = ratio 6: zero-shot for the guided model, interpolation
(not extrapolation) for conditional models trained on {2, 4, 8}.

For each requested lead this saves, in --out:
  fcst_<lead>h.npy   (T, M, C, h, w) forecast on the project's ratio-6 coarse
                     grid (block-mean grid of the ERA5 band), float32; missing
                     channels are zero-filled and marked unobserved in meta.
  truth_<lead>h.npy  (T, C, H, W) ERA5 at the valid times, on the (trimmed)
                     fine band grid.
  meta_<lead>h.npz   valid times, channels, observed mask, grids, ratio.

Run (login node):
    python -m data.download_forecast --config config/wb2_20var.yaml \
        --dataset hres --init-years 2016 2017 --leads 24 72 120 240
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config  # noqa: E402

try:  # multi-variable branches
    from utils import channel_labels, channel_specs
except ImportError:  # single-variable branches: legacy data.variable/level
    def channel_specs(dcfg):
        return dcfg.get("variables") or [{"name": dcfg["variable"],
                                          "level": dcfg.get("level")}]

    def channel_labels(dcfg):
        return [s["name"] if s.get("level") is None
                else f"{s['name']}{s['level']}" for s in channel_specs(dcfg)]

STORES = {
    "hres": ("gs://weatherbench2/datasets/hres/"
             "2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr"),
    "neuralgcm_ens": ("gs://weatherbench2/datasets/neuralgcm_ens/"
                      "2020-240x121_equiangular_with_poles_conservative.zarr"),
}


def coarse_grid(lat: np.ndarray, lon: np.ndarray, ratio: int) -> tuple:
    """Block-mean coarse grid of the fine ERA5 band grid at `ratio` (trims
    trailing rows/cols to multiples, matching coarsen())."""
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


def observed_mask(specs: list, store_vars: set) -> np.ndarray:
    """Bool per config channel: does the store carry this variable?"""
    return np.array([s["name"] in store_vars for s in specs], dtype=bool)


def _open(zarr_path: str, timeout):
    import xarray as xr
    storage = {"token": "anon"}
    if timeout:
        storage["requests_timeout"] = timeout
    try:
        return xr.open_zarr(zarr_path, storage_options=storage)
    except TypeError:
        storage.pop("requests_timeout", None)
        return xr.open_zarr(zarr_path, storage_options=storage)


def _regrid(da, lat_c: np.ndarray, lon_c: np.ndarray):
    """Bilinear interp onto the coarse grid, longitude wraparound padded."""
    import xarray as xr
    lon_n = "longitude" if "longitude" in da.dims else "lon"
    lat_n = "latitude" if "latitude" in da.dims else "lat"
    pad = da.isel({lon_n: 0})
    pad = pad.assign_coords({lon_n: float(da[lon_n][0]) + 360.0})
    da = xr.concat([da, pad], dim=lon_n)
    out = da.interp({lat_n: lat_c, lon_n: lon_c % 360.0}, method="linear")
    # interp preserves the SOURCE dim order, and the HRES store is
    # (time, longitude, latitude) — pin (lat, lon) so the memmap assignment
    # below cannot silently transpose the field.
    return out.transpose(..., lat_n, lon_n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--dataset", choices=sorted(STORES), default="hres")
    ap.add_argument("--zarr", default=None, help="override the store path")
    ap.add_argument("--leads", type=int, nargs="+", default=[24, 72, 120, 240],
                    help="lead times in hours (multiples of 24 keep the valid "
                         "hour fixed at the init hour)")
    ap.add_argument("--init-years", type=int, nargs="+", default=None,
                    help="restrict init times to these years "
                         "(default: hres -> 2016 2017 = the test years; "
                         "neuralgcm_ens -> 2020)")
    ap.add_argument("--init-stride", type=int, default=14,
                    help="take every Nth init (00 UTC only); 14 with 2 inits/"
                         "day = weekly")
    ap.add_argument("--members", type=int, default=1,
                    help="ensemble members to keep (neuralgcm_ens has 50)")
    ap.add_argument("--ratio", type=int, default=6,
                    help="1.5 deg on the 0.25 deg grid = 6")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <raw_dir>/../forecast_<dataset>)")
    ap.add_argument("--no-truth", action="store_true",
                    help="skip the ERA5 truth pull (e.g. it already exists)")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs = channel_specs(cfg["data"])
    labels = channel_labels(cfg["data"])
    fine_lat, fine_lon = load_fine_grid(cfg)
    lat_c, lon_c = coarse_grid(fine_lat, fine_lon, args.ratio)
    hf_h, hf_w = len(lat_c) * args.ratio, len(lon_c) * args.ratio
    out_dir = ensure_dir(args.out or (Path(cfg["paths"]["raw_dir"]).parent
                                      / f"forecast_{args.dataset}"))

    ds = _open(args.zarr or STORES[args.dataset], args.timeout)
    obs = observed_mask(specs, set(ds.data_vars))
    print(f"{args.dataset}: {int(obs.sum())}/{len(specs)} channels observed; "
          f"masked: {[l for l, o in zip(labels, obs) if not o] or 'none'}")

    # ── init selection: 00 UTC only, optional year filter, strided ────────
    times = ds.time.values
    hours = (times.astype("datetime64[h]")
             - times.astype("datetime64[D]")).astype(int)
    keep = hours == 0
    years = args.init_years or ([2016, 2017] if args.dataset == "hres" else [2020])
    yr = times.astype("datetime64[Y]").astype(int) + 1970
    keep &= np.isin(yr, years)
    init_idx = np.flatnonzero(keep)[::args.init_stride]
    inits = times[init_idx]
    print(f"{len(inits)} inits ({inits[0]} .. {inits[-1]}), "
          f"leads {args.leads}h, {args.members} member(s)")

    n_real = int(ds.sizes.get("realization", 1))
    m = min(args.members, n_real)

    era5 = None
    if not args.no_truth:
        era5 = _open(cfg["data"].get(
            "era5_zarr",
            "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"),
            args.timeout)

    for lead in args.leads:
        td = np.timedelta64(lead, "h")
        valid = inits + td
        fc_path = out_dir / f"fcst_{lead}h.npy"
        tr_path = out_dir / f"truth_{lead}h.npy"
        fc = np.lib.format.open_memmap(
            fc_path, mode="w+", dtype=np.float32,
            shape=(len(inits), m, len(specs), len(lat_c), len(lon_c)))

        for c, spec in enumerate(specs):
            if not obs[c]:
                fc[:, :, c] = 0.0
                continue
            da = ds[spec["name"]].sel(prediction_timedelta=td)
            if spec.get("level") is not None:
                da = da.sel(level=spec["level"])
            if "realization" in da.dims:
                da = da.isel(realization=slice(0, m))
            print(f"  [{lead:4d}h] fcst {labels[c]}", flush=True)
            for i, t0 in enumerate(inits):
                block = _regrid(da.sel(time=t0), lat_c, lon_c)
                vals = np.asarray(block.values, dtype=np.float32)
                fc[i, :, c] = vals if vals.ndim == 3 else vals[None]
        fc.flush()

        if era5 is not None:
            tr = np.lib.format.open_memmap(
                tr_path, mode="w+", dtype=np.float32,
                shape=(len(inits), len(specs), hf_h, hf_w))
            for c, spec in enumerate(specs):
                da = era5[spec["name"]]
                if spec.get("level") is not None:
                    da = da.sel(level=spec["level"])
                print(f"  [{lead:4d}h] truth {labels[c]}", flush=True)
                for i, tv in enumerate(valid):
                    field = da.sel(time=tv).sel(
                        latitude=fine_lat[:hf_h], longitude=fine_lon[:hf_w])
                    field = field.transpose(..., "latitude", "longitude")
                    tr[i, c] = np.asarray(field.values, dtype=np.float32)
            tr.flush()

        np.savez(out_dir / f"meta_{lead}h.npz",
                 valid=np.array([str(v) for v in valid]),
                 inits=np.array([str(t) for t in inits]),
                 channels=np.array(labels), observed=obs,
                 lat_coarse=lat_c, lon_coarse=lon_c,
                 lat_fine=fine_lat[:hf_h], lon_fine=fine_lon[:hf_w],
                 ratio=np.array([args.ratio]), members=np.array([m]),
                 dataset=np.array([args.dataset]))
        print(f"  [{lead:4d}h] saved -> {fc_path.name}"
              f"{'' if era5 is None else ', ' + tr_path.name}, meta")

    print(f"done -> {out_dir}")
    print("next: python -m eval.downscale_forecast --config <cfg> "
          "--ckpt <model> --data-dir " + str(out_dir))


if __name__ == "__main__":
    main()
