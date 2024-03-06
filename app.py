from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import csv
import joblib

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load('XGBoost_model.joblib')

# Configuration for serving static files
app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def hello_world():
    return render_template("exercise_recommend.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Ensure all form fields are filled
        if not all(request.form.values()):
            return render_template('exercise_recommend.html', error_message="Error: Please fill in all fields")

        int_features = [int(float(x)) for x in request.form.values()]
        with open('test.csv', 'w', newline='') as fp:
            a = csv.writer(fp)
            data = [['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'], int_features]
            a.writerows(data)

        df = pd.read_csv('test.csv')
        test = loaded_model.predict(df)
        steps = (test * 4.5)

        result = {
            "calories_burnt": test[0],
            "steps": steps[0]
        }

        return render_template('res.html', result=result)
    except Exception as e:
        print(e)
        return render_template('exercise_recommend.html', error_message="Error: Invalid input")

if __name__ == '__main__':
    app.run(debug=True)
