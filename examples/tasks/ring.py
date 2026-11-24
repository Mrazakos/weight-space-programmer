"""Ring classification: is the 2-D point inside the annulus 1 <= r <= 2?"""
import numpy as np

TASK_TYPE = "classification"
FEATURE_NAMES = ["x", "y"]
CLASSES = ["outside", "ring"]
DESCRIPTION = "points uniform in [-3,3]^2; label ring iff 1 <= sqrt(x^2+y^2) <= 2"


def generate(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    r = np.sqrt((X ** 2).sum(axis=1))
    y = ((r >= 1) & (r <= 2)).astype(int)
    return X, y
