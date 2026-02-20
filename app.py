from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/pred", methods=["POST"])
def predict():
    try:
        step = float(request.form["step"])
        transaction_type = request.form["type"]
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        transaction_type_encoded = encoder.transform([transaction_type])[0]

        input_data = np.array([[step,
                                transaction_type_encoded,
                                amount,
                                oldbalanceOrg,
                                newbalanceOrig,
                                oldbalanceDest,
                                newbalanceDest]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "Fraud Transaction 🚨"
        else:
            result = "Not Fraud Transaction ✅"

        return render_template("submit.html", prediction_text=result)

    except:
        return render_template("submit.html", prediction_text="Invalid Input!")

if __name__ == "__main__":
    app.run(debug=True)
