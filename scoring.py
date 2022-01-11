from contextlib import redirect_stdout

import joblib
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])

#################Function for model scoring
def score_model():
    """
    this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    :return:
    """

    # Load trained model
    model = joblib.load(os.path.join(model_path, "trainedmodel.pkl"))

    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y_true = test_data.pop("exited")
    test_data = test_data.drop(columns=["corporation"])
    # Predictions
    predictions = model.predict(test_data)

    # F1 score
    f1_score = metrics.f1_score(y_true, predictions, zero_division=1)

    with open("latestscore.txt", "w") as f:
        with redirect_stdout(f):
            print(f"{round(f1_score,4)}")

    return str(round(f1_score, 4))


if __name__ == "__main__":
    score_model()
