from flask import Flask, render_template,request
import pandas as pd
from src.pipeline.pred_pipeline import PredPipeline
from src.exception import CustomException
import sys
application = Flask(__name__)


# Creating route for index page
@application.route("/")
def index():
    return render_template("index.html")

@application.route('/predict', methods=['POST','GET'])
def predict():
    try:
        print('here')
        passengerId = request.form.get('passengerId')
        pclass = request.form.get('pclass')
        name = request.form.get('name')
        sex = request.form.get('sex')
        age = request.form.get('age')
        sibSp = request.form.get('sibSp')
        parch = request.form.get('parch')
        ticket = request.form.get('ticket')
        fare = request.form.get('fare')
        cabin = request.form.get('cabin')
        embarked = request.form.get('embarked')
        data = {
            'PassengerId': int(passengerId),
            'Pclass': int(pclass),
            'Name': name,
            'Sex': sex,
            'Age': float(age),
            'SibSp': int(sibSp),
            'Parch': int(parch),
            'Ticket': ticket,
            'Fare': float(fare),
            'Cabin': cabin,
            'Embarked': embarked
        }
        print(data)
        df = pd.DataFrame([data])


        obj = PredPipeline()
        result = obj.pred_pipeline(df)
        return (f'this is my {result}')
    except Exception as e:
        raise CustomException(e,sys)


if __name__ == "__main__":
    application.run(debug=True)
        