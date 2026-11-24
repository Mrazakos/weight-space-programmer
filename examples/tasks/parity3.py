"""3-bit parity: y = (x0 + x1 + x2) mod 2. A classic 'program it by hand' task."""
import numpy as np

TASK_TYPE = "classification"
FEATURE_NAMES = ["x0", "x1", "x2"]
CLASSES = ["even", "odd"]
DESCRIPTION = "inputs are bits in {0,1}; label is odd if an odd number of bits are 1"


def generate(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 3)).astype(float)
    y = (X.sum(axis=1) % 2).astype(int)
    return X, y
