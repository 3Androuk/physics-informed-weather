"""Prep cached ERA5 Z500 into the raw .npy format the pipeline expects.

Extracts channel 0 (Z500) from the FNO/AFNO baseline cache and writes
train.npy / test.npy / coords.npz into the test raw_dir. No download.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ensure_dir, load_config

CACHE = Path(r"C:\Users\andro\Github\physics-informed-weather\fno-afno-baselines\cache")

cfg = load_config("config/test_local.yaml")
raw = ensure_dir(cfg["paths"]["raw_dir"])

train = np.load(CACHE / "train_z500_t850_u850_v850_2014_2015.npy", mmap_mode="r")
test = np.load(CACHE / "val_z500_t850_u850_v850_2016_2016.npy", mmap_mode="r")
coords = np.load(CACHE / "coords.npz")

# channel 0 = Z500; (T, 120, 240)
z_train = np.ascontiguousarray(train[:, 0]).astype("float32")
z_test = np.ascontiguousarray(test[:, 0]).astype("float32")
np.save(raw / "train.npy", z_train)
np.save(raw / "test.npy", z_test)
np.savez(raw / "coords.npz", lat=coords["lat"], lon=coords["lon"])
print(f"train {z_train.shape}, test {z_test.shape} -> {raw}")
print(f"Z500 range: [{z_train.min():.0f}, {z_train.max():.0f}] mean {z_train.mean():.0f}")
