"""Create example datasets and a damaged pre-trained network (all offline, from scikit-learn).

    python examples/make_datasets.py

Writes into examples/data/:
  iris.csv                    150 rows, 4 features, target 'species'
  breast_cancer.csv           569 rows, 30 features, target 'diagnosis'
  diabetes.csv                442 rows, 10 features, regression target 'progression'
  breast_cancer_mlp.pkl       sklearn Pipeline(StandardScaler, MLPClassifier(32,16)) trained normally
  breast_cancer_mlp_damaged.pkl  the same network with its most important hidden units sabotaged
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = [iris.target_names[i] for i in iris.target]
df.drop(columns=["target"]).to_csv(out / "iris.csv", index=False)

bc = load_breast_cancer(as_frame=True)
df = bc.frame.copy()
df["diagnosis"] = [bc.target_names[i] for i in bc.target]
df = df.drop(columns=["target"])
df.to_csv(out / "breast_cancer.csv", index=False)

db = load_diabetes(as_frame=True)
df2 = db.frame.rename(columns={"target": "progression"})
df2.to_csv(out / "diabetes.csv", index=False)

# A normally trained network on the breast-cancer data (string labels, like the CSV).
X = df.drop(columns=["diagnosis"]).to_numpy()
y = df["diagnosis"].to_numpy()
pipe = make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=600, random_state=0))
pipe.fit(X, y)
joblib.dump(pipe, out / "breast_cancer_mlp.pkl")

# Damage it: flip the sign of the outgoing weights of the 4 most influential layer-1 units
# and add a large bias to the output, so it mostly predicts one class.
mlp = pipe.steps[-1][1]
infl = np.abs(mlp.coefs_[1]).sum(axis=1) * np.abs(mlp.coefs_[0]).sum(axis=0)
top = np.argsort(-infl)[:4]
mlp.coefs_[1][top, :] *= -1.0
mlp.intercepts_[2] += 3.0
joblib.dump(pipe, out / "breast_cancer_mlp_damaged.pkl")

clean = joblib.load(out / "breast_cancer_mlp.pkl")
print("clean   model accuracy on all rows:", round((clean.predict(X) == y).mean(), 4))
print("damaged model accuracy on all rows:", round((pipe.predict(X) == y).mean(), 4))
print("wrote", sorted(p.name for p in out.iterdir()))

# ---- images: scikit-learn digits as an image folder (8x8 grayscale PNGs in class subfolders) ----
from PIL import Image
from sklearn.datasets import load_digits

digits = load_digits()
root = out / "digits"
for i, (img, label) in enumerate(zip(digits.images, digits.target)):
    d = root / str(label)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.round(img / 16.0 * 255)), "L").save(d / f"{i:04d}.png")
print("wrote", len(digits.images), "digit images to", root)

# ---- text: synthetic product reviews with negation ("not good") ----
rng = np.random.default_rng(0)
pos = ["great", "excellent", "love", "amazing", "perfect", "good", "fantastic", "reliable", "comfortable", "happy"]
neg = ["terrible", "awful", "hate", "broken", "poor", "bad", "useless", "disappointing", "cheap", "flimsy"]
things = ["battery", "screen", "price", "design", "sound", "delivery", "quality", "size", "material", "support"]
fill = ["the", "it", "this", "honestly", "overall", "really", "and", "was", "is", "my", "product", "i", "think"]
rows = []
for _ in range(3000):
    sentiment = rng.integers(0, 2)
    words, score = [], 0
    for _ in range(rng.integers(2, 5)):
        w = rng.choice(pos if (rng.random() < 0.8) == bool(sentiment) else neg)
        negate = rng.random() < 0.15
        s = (1 if w in pos else -1) * (-1 if negate else 1)
        score += s
        words += list(rng.choice(fill, rng.integers(1, 4))) + (["not"] if negate else []) + [w, str(rng.choice(things))]
    rows.append({"review": " ".join(words), "sentiment": "positive" if score > 0 else ("negative" if score < 0 else
                 ("positive" if sentiment else "negative"))})
pd.DataFrame(rows).to_csv(out / "reviews.csv", index=False)
print("wrote reviews.csv")
