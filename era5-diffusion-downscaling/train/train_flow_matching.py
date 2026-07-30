"""Train conditional flow matching for ERA5 super-resolution.

Run: python -m train.train_flow_matching --config config/t2m.yaml
"""

from train.train_transport import run


if __name__ == "__main__":
    run("flow")
