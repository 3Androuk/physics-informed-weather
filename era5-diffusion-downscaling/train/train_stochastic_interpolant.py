"""Train a conditional stochastic interpolant for ERA5 super-resolution.

Run: python -m train.train_stochastic_interpolant --config config/t2m.yaml
"""

from train.train_transport import run


if __name__ == "__main__":
    run("stochastic_interpolant")
