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


def model_predictions():
    # Load trained model
    model = joblib.load(os.path.join(production_deployment_path, "trainedmodel.pkl"))

    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y_true = test_data.pop("exited")
    test_data = test_data.drop(columns=["corporation"])
    # Predictions
    predictions = model.predict(test_data)
    return predictions


def dataframe_summary():
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    data = data.drop(columns=["exited", "corporation"])

    summary_stats = data.agg(
        {
            "lastmonth_activity": ["min", "max", "mean", "median", "std"],
            "lastyear_activity": ["min", "max", "mean", "median", "std"],
            "number_of_employees": ["min", "max", "mean", "median", "std"],
        }
    )

    return summary_stats


def missing_data():
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    percent_missing = ((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
    missing_value_df = pd.DataFrame(
        {"column_name": data.columns, "percent_missing": percent_missing}
    )
    return missing_value_df


def execution_time():
    start_time = timeit.default_timer()
    os.system("python3 training.py")
    training_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system("python3 ingestion.py")
    ingestion_time = timeit.default_timer() - start_time

    run_times = [training_time, ingestion_time]
    return run_times


def outdated_packages_list():
    outdated_list = subprocess.check_output(
        ["python", "-m", "pip", "list", "--outdated"]
    )
    with open("outdated_package_list.txt", "wb") as f:
        f.write(outdated_list)


if __name__ == "__main__":
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
