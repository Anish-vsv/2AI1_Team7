from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Safe model loading
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pclass = int(request.form["Pclass"])
        sex = int(request.form["Sex"])
        age = float(request.form["Age"])
        sibsp = int(request.form["SibSp"])
        parch = int(request.form["Parch"])
        fare = float(request.form["Fare"])

        features = np.array([[pclass, sex, age, sibsp, parch, fare]])
        prediction = model.predict(features)

        result = "Survived" if prediction[0] == 1 else "Not Survived"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
