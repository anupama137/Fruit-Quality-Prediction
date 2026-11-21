
from flask import Flask, render_template, request
import pickle
import pandas as pd
app = Flask(__name__)
with open("fruit_model.pkl", "rb") as file:
    model = pickle.load(file)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    firmness = float(request.form["firmness"])
    color = float(request.form["color"])
    sugar = float(request.form["sugar"])
    spots = float(request.form["spots"])
    days = float(request.form["days"])

    features = pd.DataFrame([[
        firmness, color, sugar, spots, days
    ]], columns=[
        "Firmness", "Color_Score", "Sugar_Level",
        "Spots_Percentage", "Days_After_Harvest"
    ])

    result = model.predict(features)[0]

    return render_template(
        "index.html", prediction=result
    )
if __name__ == "__main__":
    app.run(debug=True)
