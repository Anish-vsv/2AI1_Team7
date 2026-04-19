from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    # 🔥 Convert Sex properly
    sex = 0 if form_data['Sex'] == 'male' else 1

    # 🔥 Convert other inputs
    pclass = int(form_data['Pclass'])
    age = float(form_data['Age'])
    sibsp = int(form_data['SibSp'])
    parch = int(form_data['Parch'])
    fare = float(form_data['Fare'])

    # 🔥 Feature Engineering (same as training)
    family_size = sibsp + parch

    # 🔥 Missing feature (VERY IMPORTANT)
    # Set default (since not in HTML)
    embarked = 0  

    # 🔥 Final input (8 features EXACT)
    final_input = np.array([[pclass, sex, age, sibsp, parch, fare, family_size, embarked]])

    # Prediction
    prediction = model.predict(final_input)

    result = "Survived" if prediction[0] == 1 else "Not Survived"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)