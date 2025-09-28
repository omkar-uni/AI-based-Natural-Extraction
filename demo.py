from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# =====================
# Training Dataset
# =====================
data = [
    [1, 99, 100, 9.2, 8.7, 0.5139664804, 0.4860335196],
    [2, 98, 100, 17.6, 16.5, 1.032258065, 0.967741935],
    [3, 97, 100, 26.2, 21.5, 1.647798742, 1.352201258],
    [4, 96, 100, 38.1, 32.43, 2.160782646, 1.839217354],
    [5, 95, 100, 45.1, 39.3, 2.671800948, 2.328199052],
    [6, 94, 100, 57.6, 46.8, 3.310344828, 2.689655172],
    [7, 93, 100, 67.02, 54.52, 3.859963798, 3.140036202],
    [8, 92, 100, 76.68, 62.23, 4.416096753, 3.583903247],
    [9, 91, 100, 86.34, 69.94, 4.972229332, 4.027770668],
    [10, 90, 100, 96, 77.65, 5.528361647, 4.471638353],
    [11, 89, 100, 105.66, 85.36, 6.08449377, 4.91550623],
    [12, 88, 100, 115.32, 93.07, 6.64062575, 5.35937425],
    [13, 87, 100, 124.98, 100.78, 7.196757619, 5.803242381],
    [14, 86, 100, 134.64, 108.49, 7.752889401, 6.247110599],
    [15, 85, 100, 144.3, 116.2, 8.309021113, 6.690978887],
]

df = pd.DataFrame(
    data,
    columns=[
        "Acetic",
        "Water",
        "Ethyl",
        "Burette_Aq",
        "Burette_Org",
        "Extract_Aq",
        "Extract_Org",
    ],
)

# Train Models
X = df[["Acetic", "Water", "Ethyl"]].values
y_bqA = df["Burette_Aq"].values
y_bqO = df["Burette_Org"].values
y_exA = df["Extract_Aq"].values
y_exO = df["Extract_Org"].values

model_bqA = LinearRegression().fit(X, y_bqA)
model_bqO = LinearRegression().fit(X, y_bqO)
model_exA = LinearRegression().fit(X, y_exA)
model_exO = LinearRegression().fit(X, y_exO)


# Prediction Function
def predict(acetic, water, ethyl):
    features = np.array([[acetic, water, ethyl]])

    bqA = model_bqA.predict(features)[0]
    bqO = model_bqO.predict(features)[0]
    exA = model_exA.predict(features)[0]
    exO = model_exO.predict(features)[0]

    return {
        "Burette_Aqueous": round(bqA, 3),
        "Burette_Organic": round(bqO, 3),
        "Extracted_Aqueous": round(exA, 6),
        "Extracted_Organic": round(exO, 6),
        "Total_Extracted": round(exA + exO, 6),
        "Total_Burette": round(bqA + bqO, 3),
    }


# =====================
# Flask Routes
# =====================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        acetic = float(request.form["acetic"])
        water = float(request.form["water"])
        ethyl = float(request.form["ethyl"])
        result = predict(acetic, water, ethyl)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
