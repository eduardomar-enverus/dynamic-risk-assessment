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
import shutil


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


def store_model_into_pickle():
    """

    :param model:
    :return:
    """

    # If trained model folder does not exist create one
    if not os.path.exists(prod_deployment_path):
        os.makedirs(
            prod_deployment_path
        )  # If ingest data directory does not exist create it

    files = ["trainedmodel.pkl", "ingestedfiles.txt", "latestscore.txt"]

    shutil.copy(os.path.join(model_path, files[0]), prod_deployment_path)
    shutil.copy(os.path.join(dataset_csv_path, files[1]), prod_deployment_path)
    shutil.copy(files[2], prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()
