from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# =========================
# LOAD SAVED MODELS & FILES
# =========================
rf_model = joblib.load("random_forest_fraud_model.pkl")
lr_model = joblib.load("logistic_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Numeric columns used during training for scaling
num_cols = ['amount', 'transaction_hour', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']

# Merchant categories from saved feature columns
merchant_categories = []
for col in feature_columns:
    if col.startswith("merchant_category_"):
        merchant_categories.append(col.replace("merchant_category_", ""))


@app.route('/')
def home():
    return render_template('index.html', merchant_categories=merchant_categories)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # =========================
        # GET FORM VALUES
        # =========================
        amount = float(request.form['amount'])
        transaction_hour = int(request.form['transaction_hour'])
        foreign_transaction = int(request.form['foreign_transaction'])
        location_mismatch = int(request.form['location_mismatch'])
        device_trust_score = float(request.form['device_trust_score'])
        velocity_last_24h = float(request.form['velocity_last_24h'])
        cardholder_age = int(request.form['cardholder_age'])
        merchant_category = request.form['merchant_category']

        # =========================
        # CREATE BLANK INPUT DATAFRAME
        # =========================
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Fill numeric/basic values
        input_df['amount'] = amount
        input_df['transaction_hour'] = transaction_hour
        input_df['foreign_transaction'] = foreign_transaction
        input_df['location_mismatch'] = location_mismatch
        input_df['device_trust_score'] = device_trust_score
        input_df['velocity_last_24h'] = velocity_last_24h
        input_df['cardholder_age'] = cardholder_age

        # Set selected merchant category column = 1
        merchant_col = f"merchant_category_{merchant_category}"
        if merchant_col in input_df.columns:
            input_df[merchant_col] = 1

        # =========================
        # SCALE ONLY NUMERIC COLUMNS
        # =========================
        input_scaled = input_df.copy()
        input_scaled[num_cols] = scaler.transform(input_scaled[num_cols])

        # =========================
        # LOGISTIC REGRESSION PREDICTION
        # =========================
        lr_pred = lr_model.predict(input_scaled)[0]
        lr_prob = lr_model.predict_proba(input_scaled)[0][1] * 100

        # =========================
        # RANDOM FOREST PREDICTION
        # =========================
        rf_pred = rf_model.predict(input_scaled)[0]
        rf_prob = rf_model.predict_proba(input_scaled)[0][1] * 100

        # Result text
        lr_result = "Fraud" if lr_pred == 1 else "Non-Fraud"
        rf_result = "Fraud" if rf_pred == 1 else "Non-Fraud"

        return render_template(
            'index.html',
            merchant_categories=merchant_categories,
            lr_result=lr_result,
            lr_prob=round(lr_prob, 4),
            rf_result=rf_result,
            rf_prob=round(rf_prob, 4)
        )

    except Exception as e:
        return render_template(
            'index.html',
            merchant_categories=merchant_categories,
            error=f"Error: {str(e)}"
        )


if __name__ == '__main__':
    app.run(debug=True)