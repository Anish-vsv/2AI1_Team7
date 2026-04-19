from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Dummy logic (replace with ML model)
    if int(data['sex']) == 1:
        prediction = "Survived"
    else:
        prediction = "Not Survived"

    return jsonify({"prediction":result})

if __name__ == "__main__":
    app.run(debug=True)