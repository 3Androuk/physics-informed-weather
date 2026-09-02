"""Whole-field reconstruction: direct inference vs tiled stitching vs bicubic.

Evaluates the patch-trained models on FULL test fields (the raw test split,
not patches), at every configured ratio, in two modes:

  * direct — the fully-convolutional model consumes the entire field at once
    (bottleneck attention sees ~40x more tokens than at training time; this
    run measures whether that train/test mismatch hurts);
  * tiled  — overlapping training-sized tiles, blended with a smooth window,
    sharing one global noise field, with a final exact block-average
    projection onto the observed coarse input (sample/full_field.py).

Evaluates whichever checkpoints exist: guided diffusion, flow matching,
stochastic interpolant, direct map — plus bicubic. Outputs metrics.json,
timing, and full-field comparison figures under results/full_field/.

Run:
    python -m eval.full_field --config config/t2m.yaml --n-fields 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_norm_stats  # noqa: E402
from data.degrade import coarsen, degrade  # noqa: E402
from eval.metrics import l2_norm, spectrum_log_l1  # noqa: E402
from sample.full_field import (_project_final, crop_to_multiple,  # noqa: E402
                               reconstruct_full_fused_diffusion,
                               reconstruct_full_fused_transport,
                               reconstruct_full_tiled_diffusion,
                               reconstruct_full_tiled_directmap,
                               reconstruct_full_tiled_transport)
from sample.reconstruct import load_diffusion, load_directmap  # noqa: E402
from sample.transport import load_transport  # noqa: E402
from utils import (ensure_dir, get_device, init_wandb, load_config,  # noqa: E402
                   run_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n-fields", type=int, default=4,
                    help="full test fields to evaluate (spread over the test period)")
    ap.add_argument("--ckpt", default="diffusion.pt")
    ap.add_argument("--flow-ckpt", default="flow_matching.pt")
    ap.add_argument("--si-ckpt", default="stochastic_interpolant.pt")
    ap.add_argument("--dm-ckpt", default="directmap.pt")
    ap.add_argument("--tile", type=int, default=None, help="tile size (default: patches.size)")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--batch-tiles", type=int, default=8)
    ap.add_argument("--modes", nargs="+", choices=["direct", "tiled", "fused"],
                    default=["direct", "tiled"],
                    help="fused = MultiDiffusion-style per-step tile fusion "
                         "(Bar-Tal et al., ICML 2023): one global chain, tile "
                         "predictions blended at EVERY step — seam-free by "
                         "construction, same compute as tiled")
    ap.add_argument("--no-project", action="store_true",
                    help="skip the final global data-consistency projection (tiled)")
    ap.add_argument("--project-steps", action="store_true",
                    help="per-step data-consistency projection INSIDE every tile "
                         "(and in direct mode): anchors all tiles' low frequencies "
                         "to the shared observation throughout sampling, "
                         "suppressing tile-scale seams at weak ratios (e.g. 8x)")
    ap.add_argument("--eta", type=float, default=None,
                    help="override sample.ddim_eta (eta>0 = stochastic DDIM; "
                         "measured a Pareto win to ~1.5 on this model)")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="reconstruct each stochastic method N times per field "
                         "and also score the ensemble mean. N=1 reproduces the "
                         "original single-sample path exactly.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the shared global noise fields")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "full_field")
    normalizer = load_norm_stats(patch_dir)
    tile = args.tile or cfg["patches"]["size"]
    eta = (cfg["sample"]["ddim_eta"] if args.eta is None
           else float(args.eta))
    print(f"ddim_eta = {eta}" + (" (override)" if args.eta is not None else ""))
    recons = {rc["ratio"]: rc for rc in cfg["sample"]["reconstructions"]}
    align = 16
    for r in recons:
        while align % r:
            align *= 2

    # ── Full test fields, spread over the test period ─────────────────────
    raw = np.load(Path(cfg["paths"]["raw_dir"]) / "test.npy", mmap_mode="r")
    n_fields = min(args.n_fields, len(raw))
    idxs = np.linspace(0, len(raw) - 1, n_fields).astype(int)
    fields = torch.from_numpy(np.array(raw[idxs], dtype=np.float32))
    if fields.dim() == 3:
        fields = fields[:, None]                       # (N, C, H, W)
    fields = crop_to_multiple(fields, align)           # e.g. 481 rows -> 480
    hf_all = normalizer.encode(fields)                 # normalized, on CPU
    print(f"{n_fields} full test fields {tuple(hf_all.shape[-3:])} | "
          f"tile {tile} overlap {args.overlap} | modes {args.modes} | device {device}")

    # ── Load whichever checkpoints exist ──────────────────────────────────
    methods = {}   # name -> ("diffusion"|"transport"|"directmap", payload)
    stems = []     # loaded checkpoint stems (identify the run, e.g. in wandb)
    p = ckpt_dir / args.ckpt
    if p.exists():
        model, diffusion, cfg_ck = load_diffusion(p, device)
        methods["Diffusion"] = ("diffusion", (model, diffusion),
                                _geo_full(cfg_ck, patch_dir, hf_all.shape[-2:], device))
        stems.append(p.stem)
        print(f"  Diffusion checkpoint: {p} (epoch {_epoch_of(p)})")
    for label, name in (("Flow matching", args.flow_ckpt),
                        ("Stochastic interpolant", args.si_ckpt)):
        p = ckpt_dir / name
        if p.exists():
            model, process, cfg_ck, method, residual = load_transport(p, device)
            mean_geo_full = None
            if residual is not None:
                label += " (residual)"
                if residual["mean_geo"]:  # geo-conditioned frozen mean
                    mean_geo_full = _geo_full({"geo": residual["mean_geo_cfg"]},
                                              patch_dir, hf_all.shape[-2:], device)
            methods[label] = ("transport",
                              (model, process, cfg_ck, method, residual, mean_geo_full),
                              _geo_full(cfg_ck, patch_dir, hf_all.shape[-2:], device))
            stems.append(p.stem)
            print(f"  {label} checkpoint: {p} (epoch {_epoch_of(p)})")
    p = ckpt_dir / args.dm_ckpt
    if p.exists():
        model, cfg_ck = load_directmap(p, device)
        methods["Direct map"] = ("directmap", (model,),
                                 _geo_full(cfg_ck, patch_dir, hf_all.shape[-2:], device))
        stems.append(p.stem)
        print(f"  Direct map checkpoint: {p} (epoch {_epoch_of(p)})")
    if not methods:
        raise FileNotFoundError(f"no checkpoints found in {ckpt_dir}")
    print(f"methods: {', '.join(methods)} + Bicubic")

    table = {}          # ratio -> method/mode -> {l2, spectrum_log_l1, seconds}
    for ratio, rc in recons.items():
        tag = f"{ratio}x"
        print(f"\n=== ratio {tag} ===")
        sums = {}       # (method, mode) -> [l2_sum, spec_sum, sec_sum]
        panels = {}     # label -> first-field reconstruction (for the figure)

        for fi in range(n_fields):
            hf = hf_all[fi:fi + 1].to(device)
            hf_phys = normalizer.decode(hf.cpu())
            coarse = coarsen(hf, ratio)
            lf = degrade(hf, ratio, rc.get("smooth_sigma", 0.0))
            lf_plain = degrade(hf, ratio)  # transport/direct-map input (no smoothing)
            if fi == 0:
                panels["Input (LF)"] = lf.cpu()
                panels["Reference"] = hf.cpu()

            h, w = hf.shape[-2:]
            bic = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)
            _score(sums, panels, "Bicubic", "-", bic.cpu(), hf_phys, normalizer, 0.0, fi)

            for name, (kind, payload, geo) in methods.items():
                # Direct map is deterministic: N members would be N copies.
                n_ens = 1 if kind == "directmap" else max(1, int(args.ensemble))
                if n_ens == 1:
                    # Original path, byte-for-byte (one generator across modes).
                    gen = torch.Generator().manual_seed(args.seed + fi)
                    for mode in args.modes:
                        t0 = time.time()
                        rec = _reconstruct(kind, payload, mode, hf, lf, lf_plain,
                                           coarse, ratio, rc, eta, tile, args, geo, gen)
                        if rec is None:
                            continue
                        _score(sums, panels, name, mode, rec.cpu(), hf_phys,
                               normalizer, time.time() - t0, fi)
                    continue
                for mode in args.modes:
                    t0 = time.time()
                    members = []
                    for e in range(n_ens):
                        gen = torch.Generator().manual_seed(args.seed + 10007 * e + fi)
                        rec = _reconstruct(kind, payload, mode, hf, lf, lf_plain,
                                           coarse, ratio, rc, eta, tile, args, geo, gen)
                        if rec is None:
                            break
                        members.append(rec)
                    if not members:
                        continue
                    sec = time.time() - t0
                    _score(sums, panels, name, mode, members[0].cpu(), hf_phys,
                           normalizer, sec / n_ens, fi)
                    _score(sums, panels, name, f"{mode} ens{n_ens}",
                           torch.stack(members).mean(0).cpu(), hf_phys, normalizer,
                           sec, fi)

        row = {}
        for (name, mode), (l2s, specs, secs) in sums.items():
            key = name if mode == "-" else f"{name} ({mode})"
            row[key] = {"l2": l2s / n_fields, "spectrum_log_l1": specs / n_fields,
                        "seconds_per_field": secs / n_fields}
            print(f"  {key:32s} | L2 {row[key]['l2']:.4f} | "
                  f"spec-logL1 {row[key]['spectrum_log_l1']:.4f} | "
                  f"{row[key]['seconds_per_field']:.1f}s/field")
        table[tag] = row
        _figure(panels, normalizer, ratio, results_dir / f"full_{tag}.png")

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nOutputs -> {results_dir}")

    # ── wandb (opt-in): one-shot eval -> summary scalars + figures ─────────
    wb_run, wandb = init_wandb(
        cfg, job_type="eval_full_field",
        extra_config={"n_fields": n_fields, "tile": tile, "overlap": args.overlap,
                      "modes": args.modes, "checkpoints": stems},
        name=run_name(cfg, "fullfield", *stems,
                      "projsteps" if args.project_steps else "",
                      "noproj" if args.no_project else ""))
    if wb_run is not None:
        tbl = wandb.Table(columns=["ratio", "method", "l2", "spectrum_log_l1",
                                   "seconds_per_field"])
        for tag, row in table.items():
            for key, v in row.items():
                tbl.add_data(tag, key, v["l2"], v["spectrum_log_l1"],
                             v["seconds_per_field"])
                skey = key.lower().replace(" ", "_").replace("(", "").replace(")", "")
                wb_run.summary[f"{tag}/{skey}/l2"] = v["l2"]
                wb_run.summary[f"{tag}/{skey}/spectrum_log_l1"] = v["spectrum_log_l1"]
        log = {"eval/full_field_table": tbl}
        for tag in table:
            fig_path = results_dir / f"full_{tag}.png"
            if fig_path.exists():
                log[f"eval/full_{tag}"] = wandb.Image(str(fig_path))
        wb_run.log(log)
        wb_run.finish()
        print("wandb: full-field eval logged")


@torch.no_grad()
def _reconstruct(kind, payload, mode, hf, lf, lf_plain, coarse, ratio, rc, eta,
                 tile, args, geo, gen):
    """One method/mode reconstruction of one full field (normalized units)."""
    project = not args.no_project
    if kind == "diffusion":
        model, diffusion = payload
        if mode == "direct":
            cond = _geo_batched(geo)
            rec = diffusion.guided_reconstruct(
                model, lf, t_steps=rc["t_steps"], K=rc["K"], eta=eta, cond=cond,
                project=args.project_steps, lf=coarse if args.project_steps else None,
                ratio=ratio if args.project_steps else None)
            return _project_final(rec, coarse, ratio) if project else rec
        if mode == "fused":
            return reconstruct_full_fused_diffusion(
                diffusion, model, lf, coarse, ratio, rc, eta=eta, tile=tile,
                overlap=args.overlap, batch=args.batch_tiles, geo_full=geo,
                project_steps=args.project_steps, project_final=project,
                generator=gen)
        return reconstruct_full_tiled_diffusion(
            diffusion, model, lf, coarse, ratio, rc, eta=eta, tile=tile,
            overlap=args.overlap, batch=args.batch_tiles, geo_full=geo,
            project_steps=args.project_steps, project_final=project, generator=gen)
    if kind == "transport":
        model, process, cfg_ck, method, residual, mean_geo_full = payload
        tc = cfg_ck.get("transport", {})
        # Residual transport checkpoints condition on the mean field and sample
        # the normalized residual: coarsen(x) == observation holds only for the
        # COMPOSED field, so any projection moves outside the sampler.
        cond_full = (lf_plain if residual is None
                     else _mean_full(hf, ratio, residual, mean_geo_full))
        inner_project = project and residual is None
        if mode == "direct":
            kwargs = dict(coords=_geo_batched(geo), steps=tc.get("sample_steps", 100),
                          solver=tc.get("solver", "heun"),
                          project="final" if inner_project else "none",
                          coarse=coarse if inner_project else None,
                          ratio=ratio if inner_project else None)
            if method == "stochastic_interpolant":
                si = tc.get("stochastic_interpolant", {})
                kwargs.update(sampler=si.get("sampler", "ode"),
                              stochasticity=si.get("stochasticity", 0.1))
            out = process.sample(model, cond_full, **kwargs)
        elif mode == "fused":
            out = reconstruct_full_fused_transport(
                model, process, cond_full, coarse, ratio, cfg_ck, method, tile=tile,
                overlap=args.overlap, batch=args.batch_tiles, geo_full=geo,
                project_final=inner_project,
                project_each=args.project_steps and residual is None,
                generator=gen)
        else:
            out = reconstruct_full_tiled_transport(
                model, process, cond_full, coarse, ratio, cfg_ck, method, tile=tile,
                overlap=args.overlap, batch=args.batch_tiles, geo_full=geo,
                project_final=inner_project,
                project_each=args.project_steps and residual is None,
                generator=gen)
        if residual is not None:
            out = cond_full + residual["res_std"] * out
            if project:
                out = _project_final(out, coarse, ratio)
        return out
    if kind == "directmap":
        (model,) = payload
        if mode == "direct":
            cond = _geo_batched(geo)
            return model(lf_plain, None, cond) if cond is not None else model(lf_plain)
        if mode == "fused":
            return None  # single-pass regression: fused would duplicate tiled
        return reconstruct_full_tiled_directmap(
            model, lf_plain, tile=tile, overlap=args.overlap,
            batch=args.batch_tiles, geo_full=geo)
    raise ValueError(kind)


def _geo_full(cfg_ck, patch_dir, hw, device):
    """Full-grid geo payload for a geo-conditioned checkpoint (else None).

    Hash coords: (H, W, d); packed HEALPix payload: (L, H, W, 8) — both cropped
    to the align-cropped field size, ready for per-tile cropping."""
    if not cfg_ck.get("geo", {}).get("enabled", False):
        return None
    g = cfg_ck["geo"]
    h, w = hw
    encoder = g.get("encoder", "hash")
    if encoder == "healpix":
        hp = np.load(patch_dir / g.get("healpix_index", "healpix_index.npz"))
        idx = torch.from_numpy(hp["idx"][:, :h, :w, :].astype(np.float32))
        wts = torch.from_numpy(np.ascontiguousarray(hp["w"][:, :h, :w, :]))
        return torch.cat([idx, wts], dim=-1).to(device)
    if encoder in ("static", "hash_static"):
        sf = np.load(patch_dir / "static_fields.npz")
        fields = np.ascontiguousarray(sf["fields"][:, :h, :w])
        static = torch.from_numpy(fields).permute(1, 2, 0)           # (h, w, S)
        if encoder == "static":
            return static.to(device)
    # hash / xyz / sinusoidal / hash_static consume the coordinates
    from models.geo_encoding import build_patch_coords
    cf = np.load(patch_dir / "coords_full.npz")
    alt = g.get("altitude") if g.get("input_dim", 3) == 4 else None
    coords = torch.from_numpy(
        build_patch_coords(cf["lat"][:h], cf["lon"][:w], altitude=alt))
    if encoder == "hash_static":
        return torch.cat([coords, static], dim=-1).to(device)        # (h, w, d+S)
    return coords.to(device)


def _geo_batched(geo):
    """Add the batch dim for a single-field direct pass."""
    return None if geo is None else geo[None]


@torch.no_grad()
def _mean_full(hf, ratio, residual, mean_geo_full):
    """Full-field deterministic mean for residual transport checkpoints:
    the frozen learned regression (full-field pass), or bicubic."""
    if residual["mean_model"] is not None:
        x = degrade(hf, ratio)
        m = residual["mean_model"]
        return (m(x, None, _geo_batched(mean_geo_full))
                if residual["mean_geo"] else m(x))
    h, w = hf.shape[-2:]
    return F.interpolate(coarsen(hf, ratio), size=(h, w), mode="bicubic",
                         align_corners=False)


def _epoch_of(ckpt_path):
    """Epoch counter stored in a checkpoint, for run identification."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ck.get("epoch", "?")
    except Exception:  # noqa: BLE001 - identification only, never fatal
        return "?"


def _score(sums, panels, name, mode, rec, hf_phys, normalizer, seconds, field_idx):
    rec_phys = normalizer.decode(rec)
    entry = sums.setdefault((name, mode), [0.0, 0.0, 0.0])
    entry[0] += l2_norm(rec_phys, hf_phys)
    entry[1] += spectrum_log_l1(rec_phys, hf_phys)
    entry[2] += seconds
    if field_idx == 0:
        panels[name if mode == "-" else f"{name} ({mode})"] = rec


def _figure(panels, normalizer, ratio, path):
    """Stacked full-field panels (first test field), shared color scale."""
    ref = normalizer.decode(panels["Reference"])[0, 0].numpy()
    vmin, vmax = float(ref.min()), float(ref.max())
    names = [n for n in panels if n != "Reference"] + ["Reference"]
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3.2 * len(names)))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        ax.imshow(normalizer.decode(panels[name])[0, 0].numpy(), cmap="RdBu_r",
                  vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{ratio}x full-field reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
