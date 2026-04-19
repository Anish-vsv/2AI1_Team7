from flask import Flask, request, jsonify, render_template
import numpy as np
# import joblib  # Uncomment when using real model

app = Flask(__name__)

# Load your trained model (optional)
# model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get values from frontend
        pclass = int(data["pclass"])
        sex = int(data["sex"])
        age = float(data["age"])
        sibsp = int(data["sibsp"])
        parch = int(data["parch"])
        fare = float(data["fare"])

        # Convert into array for model
        features = np.array([[pclass, sex, age, sibsp, parch, fare]])

        # 🔥 Replace this with your ML model prediction
        # prediction = model.predict(features)[0]

        # Dummy logic (for testing)
        if sex == 1:
            prediction = 1
        else:
            prediction = 0

        result = "Survived" if prediction == 1 else "Not Survived"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})


# IMPORTANT for Render deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)