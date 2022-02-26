from flask import Flask
from flask import request
from flask import render_template
import joblib

app = Flask(__name__)
lr_model = joblib.load("CCU_LR")
dt_model = joblib.load("CCU_DT")
mlp_model = joblib.load("CCU_MLP")
rf_model = joblib.load("CCU_RF")
gb_model = joblib.load("CCU_GB")

# @ is a function decorator
# must run the app.route first before running any function below


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        purchases = request.form.get('purchases')
        print(purchases)
        supp_cards = request.form.get('supp_cards')
        print(supp_cards)
        lr_pred = lr_model.predict([[float(purchases), supp_cards]])
        print("lr is " + str(lr_pred[0]))

        dt_pred = dt_model.predict([[float(purchases), supp_cards]])
        print("dt is " + str(dt_pred[0]))

        mlp_pred = mlp_model.predict([[float(purchases), supp_cards]])
        print("mlp is " + str(mlp_pred[0]))

        rf_pred = rf_model.predict([[float(purchases), supp_cards]])
        print("rf is " + str(rf_pred[0]))

        gb_pred = gb_model.predict([[float(purchases), supp_cards]])
        print("gb is " + str(gb_pred[0]))

        if lr_pred[0] == 0:
            result1 = "No Upgrade"
        elif lr_pred[0] == 1:
            result1 = "Upgrade"

        if dt_pred[0] == 0:
            result2 = "No Upgrade"
        elif dt_pred[0] == 1:
            result2 = "Upgrade"

        if mlp_pred[0] == 0:
            result3 = "No Upgrade"
        elif mlp_pred[0] == 1:
            result3 = "Upgrade"

        if rf_pred[0] == 0:
            result4 = "No Upgrade"
        elif rf_pred[0] == 1:
            result4 = "Upgrade"

        if gb_pred[0] == 0:
            result5 = "No Upgrade"
        elif gb_pred[0] == 1:
            result5 = "Upgrade"

        return (render_template("index.html", result1='Logistic Regression model predicts: ' + result1, result2='Decision Tree model predicts: ' + result2, result3='Neural Network model predicts: ' + result3, result4='Random Forest model predicts: ' + result4, result5='Gradient Boosted Decision Tree model predicts: ' + result5))
    else:
        return (render_template("index.html", result1='No input submitted. Please submit a rate!', result2='No input submitted. Please submit a rate!', result3='No input submitted. Please submit a rate!', result4='No input submitted. Please submit a rate!', result5='No input submitted. Please submit a rate!'))
