from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ==========================
# Load saved objects
# ==========================

model = joblib.load("best_salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    try:
        # ===== Numerical inputs =====
        years_experience = float(request.form['years_experience'])
        remote_ratio = float(request.form['remote_ratio'])
        benefits_score = float(request.form['benefits_score'])
        days_open = float(request.form['days_open'])
        num_skills = float(request.form['num_skills'])

        # ===== Categorical inputs =====
        job_title = request.form['job_title']
        company_location = request.form['company_location']
        company_size = request.form['company_size']
        employee_residence = request.form['employee_residence']
        education_required = request.form['education_required']
        industry = request.form['industry']
        salary_currency = request.form['salary_currency']

        # ==========================
        # Feature engineering
        # ==========================
        input_dict = {
            'years_experience': years_experience,
            'remote_ratio': remote_ratio,
            'benefits_score': benefits_score,
            'num_skills': num_skills,
            'exp_x_skills': years_experience * num_skills,
            'benefits_x_exp': benefits_score * years_experience,
            'same_country': 1
    
        }

        # ==========================
        # Encoding
        # ==========================
        for col in label_encoders:
            try:
                encoded = label_encoders[col].transform([locals()[col]])[0]
            except:
                encoded = 0  # unseen category fallback
            input_dict[f"{col}_encoded"] = encoded

        # ===== Boolean placeholders =====
        input_dict.update({
            'emp_full_time': 1,
            'emp_part_time': 0,
            'emp_freelance': 0,
            'remote_onsite': 0,
            'remote_remote': 1
        })

        input_df = pd.DataFrame([input_dict])

        scaled = scaler.transform(input_df)
        prediction_log = model.predict(scaled)
        prediction = np.expm1(prediction_log)[0]

        return render_template("index.html", prediction_text=f"Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
