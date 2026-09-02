"""The cost-benefit question: HRES at 0.25 deg vs HRES at 1.5 deg + downscaling.

The 1.5 deg product IS a conservative regrid of the 0.25 deg one, so the
native field is the CEILING our reconstruction is trying to recover. Scoring
it against the same ERA5 truth, on the same valid times, band and metric as
eval/downscale_forecast gives the missing reference:

    native      = HRES 0.25 deg              (expensive path)
    downscaled  = model(HRES 1.5 deg)        (cheap path, already measured)
    bicubic     = bicubic(HRES 1.5 deg)      (cheap path, no model)

gap = downscaled - native  is what dissemination at 1.5 deg costs you.
If it is small, the coarse product plus downscaling is a viable substitute.

Also reports MAE, because CRPS of a DETERMINISTIC field reduces to its MAE --
that is the correct baseline for the ensemble CRPS numbers, which were being
compared against RMSE.

Run:  python score_native_hres.py --data-dir datasets/forecast_hres_t2m --lead 24
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

NATIVE = ("gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr")


def latw(lat_deg):
    w = np.cos(np.deg2rad(lat_deg)).astype(np.float64)
    return w / w.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lead", type=int, required=True)
    ap.add_argument("--limit", type=int, default=8,
                    help="must match the downscaling run being compared")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.data_dir)
    meta = np.load(d / f"meta_{args.lead}h.npz")
    truth = np.load(d / f"truth_{args.lead}h.npy", mmap_mode="r")
    fcst_c = np.load(d / f"fcst_{args.lead}h.npy", mmap_mode="r")
    valid = meta["valid"].astype("datetime64[ns]")
    inits = meta["inits"].astype("datetime64[ns]")
    lat_f, lon_f = meta["lat_fine"], meta["lon_fine"]
    ratio = int(meta["ratio"][0])
    n = min(args.limit, len(inits))
    print(f"{args.lead}h | {n} inits | band {len(lat_f)}x{len(lon_f)} | ratio {ratio}")

    ds = xr.open_zarr(NATIVE, storage_options={"token": "anon"},
                      decode_timedelta=True)
    td = np.timedelta64(args.lead, "h")
    da = ds["2m_temperature"].sel(prediction_timedelta=td)

    w = latw(lat_f)[None, :, None]
    acc = {k: 0.0 for k in ("native_se", "native_ae", "bicub_se", "bicub_ae")}
    npix = 0
    for i in range(n):
        t_true = np.asarray(truth[i], dtype=np.float64)            # (C,H,W)
        native = da.sel(time=inits[i]).sel(
            latitude=lat_f, longitude=lon_f).transpose(
            "latitude", "longitude").values.astype(np.float64)[None]

        # bicubic of the SAME coarse field the model consumed
        import torch
        import torch.nn.functional as F
        c = torch.from_numpy(np.asarray(fcst_c[i, 0], dtype=np.float32))[None]
        bic = F.interpolate(c, size=t_true.shape[-2:], mode="bicubic",
                            align_corners=False)[0].numpy().astype(np.float64)

        for tag, pred in (("native", native), ("bicub", bic)):
            e = pred - t_true
            acc[f"{tag}_se"] += float((w * e ** 2).sum())
            acc[f"{tag}_ae"] += float((w * np.abs(e)).sum())
        npix += t_true.size

    out = {
        "lead_h": args.lead, "n_inits": n, "source": "hres_native_0p25",
        "native_rmse_latweighted": float(np.sqrt(acc["native_se"] / npix)),
        "native_mae_latweighted": float(acc["native_ae"] / npix),
        "bicubic_rmse_latweighted": float(np.sqrt(acc["bicub_se"] / npix)),
        "bicubic_mae_latweighted": float(acc["bicub_ae"] / npix),
    }
    print(json.dumps(out, indent=2))
    p = Path(args.out or (d.parent.parent / "results_t2m" /
                          f"native_hres_{args.lead}h.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
