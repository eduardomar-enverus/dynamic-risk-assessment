import glob
import json
import os
import re

import pandas as pd
from sklearn.metrics import f1_score

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

with open("./config.json", "r") as file:
    config = json.load(file)

input_folder_path = config["input_folder_path"]
dataset_path = config["output_folder_path"]
prod_deployment_path = config["prod_deployment_path"]
deployed_score = os.path.join(prod_deployment_path, "latestscore.txt")
deployed_ingested_files = os.path.join(prod_deployment_path, "ingestedfiles.txt")


def find_csv_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            files.append(file)
    return files


def check_drift(dataframe):
    """
    check_drift function is used to check if the model is drifted or not
    Args:
        dataframe (pd.DataFrame): dataframe containing the data
    Returns:
        boolean: True if the model is drifted, False otherwise
    """

    # Reading ingestedfiles.txt from production deployment folder
    with open(deployed_ingested_files) as files:
        ingested_files = files.readlines()
        ingested_files = [line.rstrip() for line in ingested_files]

    # Check whether source files are same as ingested files
    source_files = find_csv_files(input_folder_path)

    if set(ingested_files) == set(source_files):
        return False

    else:
        # Ingesting new data
        ingestion.merge_multiple_dataframe()

        # Checking for model drift
        with open(deployed_score) as score_file:
            stored_score = score_file.read()
            stored_score = float(stored_score)

        label = dataframe["exited"]
        features = dataframe.drop(["exited", "corporation"], axis=1)

        y_pred = diagnostics.model_predictions(features)
        new_score = f1_score(label.values, y_pred)

        # Printing new vs stored scores
        print("Stored score = %s", stored_score)
        print("New score = %s", new_score)

        # Check if model drifting happened
        if new_score < stored_score:  # if new score is greater than deployed score
            print("Drift occurred")
            return True
        else:
            print("No drift")
            return False


def retrain():
    print("Retraining")
    training.train_model()

    print("Rescoring")
    scoring.score_model()

    print("Redeploying")
    deployment.store_model_into_pickle()

    print("Running API calls and reporting")
    os.system("python reporting.py")
    os.system("python apicalls.py")


if __name__ == "__main__":
    input_dataframe = pd.read_csv(os.path.join(dataset_path, "finaldata.csv"))

    DRIFT_CHECK = check_drift(input_dataframe)
    if DRIFT_CHECK is True:
        retrain()
