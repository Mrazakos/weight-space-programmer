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
