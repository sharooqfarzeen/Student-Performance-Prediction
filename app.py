import os
from flask import Flask, render_template, request, redirect, url_for
from src.pipeline.predict_pipeline import CustomData, Predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit():
    # Get form data and convert it into a pandas dataframe
    new_data = CustomData(
    writing_score = request.form['writing_score'],
    reading_score = request.form['reading_score'],
    gender = request.form['gender'],
    race_ethnicity = request.form['race_ethnicity'],
    parental_level_of_education = request.form['parental_level_of_education'],
    lunch = request.form['lunch'],
    test_preparation_course = request.form['test_preparation_course']
    )

    df = new_data.get_dataframe()

    #getting prediction
    prediction = Predict().predict(df)
    #accounting for score greater than 100
    score = min(100, prediction[0])
    
    # return redirect(url_for('result', math_score = score))
    return render_template('result.html', score = score)

if __name__ == '__main__':
    app.run(debug=True)