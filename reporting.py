import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.metrics import plot_confusion_matrix


from diagnostics import model_predictions

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])


def score_model():


    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    predictions = model_predictions(test_data.drop(columns=["corporation","exited"]))

    y_true = test_data.pop("exited")

    data = {"y_Actual": y_true, "y_Predicted": predictions}

    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(
        df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
    )

    sns.heatmap(confusion_matrix, annot=True)
    plt.show()
    plt.savefig(os.path.join(output_model_path,'confusionmatrix.png'))


if __name__ == "__main__":
    score_model()
