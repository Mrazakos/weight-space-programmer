"""Text task: sentiment of short phrases where 'not' flips the next sentiment word."""
import numpy as np

MODALITY = "text"
TASK_TYPE = "classification"
CLASSES = ["negative", "positive"]
DESCRIPTION = "label is positive iff the sum of word polarities is > 0, where 'not' flips the following word"

POS = ["good", "great", "nice", "fine", "excellent"]
NEG = ["bad", "awful", "poor", "terrible", "boring"]
FILL = ["the", "movie", "was", "plot", "acting", "really", "quite", "and", "but"]


def generate(n, seed):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    while len(texts) < n:
        words, score = [], 0
        for _ in range(rng.integers(1, 4)):
            w = rng.choice(POS + NEG)
            flip = rng.random() < 0.3
            score += (1 if w in POS else -1) * (-1 if flip else 1)
            words += list(rng.choice(FILL, rng.integers(0, 3))) + (["not"] if flip else []) + [w]
        if score == 0:
            continue
        texts.append(" ".join(words))
        labels.append(int(score > 0))
    return texts, labels
