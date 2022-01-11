import json
import os

import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:5000"

with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = config["test_data_path"]
prediction_file = os.path.join(test_data_path, "testdata.csv")
output_model_path = config["output_model_path"]

api_returns = os.path.join(output_model_path, "apireturns.txt")

with open(api_returns, "w") as file:
    file.write("API Returns Data: \n")

    file.write("Data Statistics Summary\n")
    file.write(requests.get(f"{URL}/summarystats").text)
    file.write("\n")

    file.write("Diagnostics\n")
    file.write(requests.get(f"{URL}/diagnostics").text)
    file.write("\n")

    file.write("Model Predictions on Test Data\n")
    file.write(
        requests.post(
            f"{URL}/prediction", json={"prediction_file_path": prediction_file}
        ).text
    )
    file.write("\n")

    file.write("Model Score: " + requests.get(f"{URL}/scoring").text)
