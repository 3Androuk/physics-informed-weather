"""Head-to-head vs the sibling era5-diffusion-downscaling full-field eval.

Scores the DLWP-HPX SR model on the SAME benchmark as the sibling project's
eval/full_field.py (the tiled / fused diffusion reconstructions): the same
test timestamps (np.linspace over the test split), the same +-60 deg 0.25 deg
lat-lon grid (rows from the sibling raw download's coords.npz), the same
crop_to_multiple row crop, and verbatim copies of its l2_norm /
spectrum_log_l1 metrics, in physical units.

Pipeline per field: HPX truth faces -> degrade (exact HEALPix coarsen 4x)
-> model -> decode to K -> hpx_to_latlon onto the band grid -> crop -> score
against the true ERA5 field fetched from the WB2 store (NOT a remap of the
mesh truth), exactly what the sibling used as reference.

Also reported:
  * Bicubic recomputed here the sibling way (avg-pool 4x + bicubic upsample
    of the true lat-lon field) — must reproduce the sibling's recorded
    Bicubic row, validating grid/crop/metric conventions;
  * the remap floor: HPX truth faces remapped to the band grid and scored —
    the error the mesh->grid interpolation alone adds to our model's row
    (the sibling models pay no such penalty).

Run (needs the sibling raw coords, not its deleted raw fields):
    python -m eval.compare_full_field --config config/default.yaml \
        --raw-coords ../era5-diffusion-downscaling/datasets/raw_t2m/coords.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import Normalizer  # noqa: E402
from data.degrade import degrade_faces  # noqa: E402
from data.download_era5 import _open_da  # noqa: E402
from hpx.remap import hpx_to_latlon  # noqa: E402
from models.hpx_unet import build_model  # noqa: E402
from utils import ensure_dir, get_device, load_config  # noqa: E402


# ── verbatim from the sibling eval/metrics.py (keep numerics identical) ────
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:        # (N, C, H, W) -> assume single channel
        x = x[:, 0]
    elif x.ndim == 2:      # (H, W) -> (1, H, W)
        x = x[None]
    return x               # (N, H, W)


def l2_norm(pred, truth) -> float:
    """Mean over samples of per-sample RMSE = sqrt(mean_grid (pred - truth)^2)."""
    p, t = _to_numpy(pred), _to_numpy(truth)
    per_sample = np.sqrt(((p - t) ** 2).mean(axis=(-2, -1)))
    return float(per_sample.mean())


def radial_power_spectrum(fields):
    f = _to_numpy(fields)
    n, h, w = f.shape
    fhat = np.fft.fftshift(np.fft.fft2(f, axes=(-2, -1)), axes=(-2, -1))
    psd = (np.abs(fhat) ** 2) / (h * w)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.round(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(int)
    kmax = min(cy, cx)
    E = np.empty((n, kmax + 1))
    for k in range(kmax + 1):
        E[:, k] = psd[:, r == k].mean(axis=1)
    return np.arange(kmax + 1), E.mean(axis=0)


def spectrum_log_l1(pred, truth) -> float:
    """Mean absolute log10-spectrum error over wavenumbers k >= 1 (skip DC)."""
    _, ep = radial_power_spectrum(pred)
    _, et = radial_power_spectrum(truth)
    eps = 1e-30
    return float(np.mean(np.abs(np.log10(ep[1:] + eps) - np.log10(et[1:] + eps))))
# ───────────────────────────────────────────────────────────────────────────


def crop_to_multiple(x: np.ndarray, m: int) -> np.ndarray:
    """Sibling sample/full_field.py: crop trailing (H, W) to multiples of m."""
    h, w = x.shape[-2:]
    return x[..., : (h // m) * m, : (w // m) * m]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--raw-coords", required=True,
                    help="coords.npz of the sibling raw band download "
                         "(defines the exact +-60 deg rows and their order)")
    ap.add_argument("--n-fields", type=int, default=4,
                    help="must match the sibling full_field run being compared")
    ap.add_argument("--align", type=int, default=16,
                    help="crop_to_multiple alignment used by the sibling run")
    ap.add_argument("--checkpoint", default=None,
                    help="override eval.checkpoint (default best.pt)")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    ratio = int(cfg["sr"]["ratio"])
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / (
        args.checkpoint or cfg["eval"]["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = Normalizer(ckpt["norm_mean"], ckpt["norm_std"])
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']})")

    test = np.load(hpx_dir / "test.npy", mmap_mode="r")
    times = np.load(hpx_dir / "test_times.npy")
    idxs = np.linspace(0, len(test) - 1, args.n_fields).astype(int)
    print(f"fields {list(idxs)} of {len(test)} | times {times[idxs]}")

    # ── True ERA5 band fields for those timestamps, from the store ────────
    coords = np.load(args.raw_coords)
    band_lat, band_lon = coords["lat"].astype(np.float64), coords["lon"].astype(np.float64)
    da = _open_da(cfg["data"], args.timeout, chunk_time=1)  # lat ascending
    st_lat = da["latitude"].values.astype(np.float64)
    lat_asc = band_lat if band_lat[0] < band_lat[-1] else band_lat[::-1]
    row_idx = np.searchsorted(st_lat, lat_asc)
    if not np.allclose(st_lat[row_idx], lat_asc, atol=1e-4):
        raise ValueError("raw-coords latitudes are not rows of the store grid")
    sub = da.sel(time=times[idxs]).isel(latitude=row_idx)
    truth = sub.values.astype(np.float32)          # (N, H_band, W) ascending lat
    if band_lat[0] > band_lat[-1]:
        truth = truth[:, ::-1]                     # back to the raw row order
    print(f"truth fetched: {truth.shape} in K")

    # ── DLWP-HPX prediction on the same timestamps ────────────────────────
    y_faces = np.asarray(test[idxs], dtype=np.float32)          # (N, 12, F, F) K
    y = normalizer.encode(torch.from_numpy(y_faces))[:, :, None]  # (N,12,1,F,F)
    with torch.no_grad():
        preds = []
        for i in range(len(idxs)):                # batch 1: modest VRAM
            x = degrade_faces(y[i:i + 1].to(device), ratio)
            preds.append(model(x).cpu())
        pred = torch.cat(preds)
    pred_K = normalizer.decode(pred)[:, :, 0].numpy()            # (N, 12, F, F)

    print("remapping mesh -> band grid ...")
    pred_ll = hpx_to_latlon(pred_K, band_lat, band_lon)          # (N, H_band, W)
    floor_ll = hpx_to_latlon(y_faces, band_lat, band_lon)        # remap floor

    # ── Sibling-style bicubic from the TRUE lat-lon field ─────────────────
    t = torch.from_numpy(np.ascontiguousarray(crop_to_multiple(truth, args.align)))[:, None]
    bic = F.interpolate(F.avg_pool2d(t, ratio), size=t.shape[-2:],
                        mode="bicubic", align_corners=False)[:, 0].numpy()

    truth_c = crop_to_multiple(truth, args.align)
    rows = {
        "DLWP-HPX (remapped)": crop_to_multiple(pred_ll, args.align),
        "Bicubic (validation)": bic,
        "HPX truth remap floor": crop_to_multiple(floor_ll, args.align),
    }
    out = {"n_fields": int(args.n_fields), "field_idxs": [int(i) for i in idxs],
           "grid": list(truth_c.shape[-2:]), "ratio": ratio}
    for name, rec in rows.items():
        out[name] = {"l2": l2_norm(rec, truth_c),
                     "spectrum_log_l1": spectrum_log_l1(rec, truth_c)}
        print(f"{name:24s} l2 {out[name]['l2']:.4f} K | "
              f"spec {out[name]['spectrum_log_l1']:.4f}")

    path = Path(results_dir) / "compare_full_field.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
