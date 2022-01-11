import subprocess
from datetime import time

import joblib
import pandas as pd
import numpy as np
import timeit
import os
import json


with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
production_deployment_path = os.path.join(config["prod_deployment_path"])

# Load test data
test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
y_true = test_data.pop("exited")
test_data = test_data.drop(columns=["corporation"])

def model_predictions(test_data):
    # Load trained model
    model = joblib.load(os.path.join(production_deployment_path, "trainedmodel.pkl"))

    # Predictions
    predictions = model.predict(test_data)
    return predictions


def dataframe_summary():
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    data_numbers = data.drop(
        ['exited'], axis=1).select_dtypes('number')

    statistics_dict = {}
    for column in data_numbers.columns:
        mean = data_numbers[column].mean()
        median = data_numbers[column].median()
        std = data_numbers[column].std()

        statistics_dict[column] = {
            'mean': round(mean, 3),
            'median': round(median, 3),
            'std': round(std, 3)}

    return statistics_dict


def missing_data():
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    missing_list = {col: {'percentage': percentage} for col, percentage in zip(
        data.columns, data.isna().sum() / data.shape[0] * 100)}
    return missing_list


def execution_time():
    start_time = timeit.default_timer()
    os.system("python3 training.py")
    training_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system("python3 ingestion.py")
    ingestion_time = timeit.default_timer() - start_time

    run_times = [
        {'ingestion_time': round(ingestion_time, 3)},
        {'training_time': round(training_time, 3)}
    ]
    return run_times


def outdated_packages_list():
    dep = subprocess.run(
       [ 'pip', 'list', '--outdated'],stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8').stdout
    return dep


if __name__ == "__main__":
    # model_predictions()
    # dataframe_summary()
    # missing_data()
    # execution_time()
    # outdated_packages_list()

    print("Outdated Packages")
    print(outdated_packages_list())
