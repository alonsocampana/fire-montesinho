#adapted from https://www.statworx.com/at/blog/how-to-build-a-machine-learning-api-with-python-and-flask/
from flask import Flask
from flask_restful import Api, Resource, reqparse
import pandas as pd
import sys
sys.path.append("../imports")
from model_end_to_end import *
import pickle
import numpy as np

APP = Flask(__name__)
API = Api(APP)

with open("model_gbr_re.pickle", 'rb') as f:
    MONTESINHO_MODEL = pickle.load(f)

class Predict(Resource):

    @staticmethod
    def post():
        #MONTESINHO_MODEL.train()
        parser = reqparse.RequestParser()
        parser.add_argument('X', type=int)
        parser.add_argument('Y', type=int)
        parser.add_argument('month')
        parser.add_argument('day')
        parser.add_argument('FFMC',type=float)
        parser.add_argument('DMC',type=float)
        parser.add_argument('DC', type=float)
        parser.add_argument('ISI',type=float)
        parser.add_argument('temp',type=float)
        parser.add_argument('RH',type=float)
        parser.add_argument('wind',type=float)
        parser.add_argument('rain',type=float)

        args = parser.parse_args()  # creates dict
        X_new = pd.Series(args)
        out = {'Prediction': MONTESINHO_MODEL.predict_instance(X_new)}
        return out, 200

API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')
