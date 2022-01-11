import pandas as pd
from flask import Flask, jsonify, request
from scoring import score_model
from diagnostics import (
    outdated_packages_list,
    execution_time,
    missing_data,
    dataframe_summary,
    model_predictions,
)

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"


@app.route("/")
def index():
    """
    Route Endpoint
    """
    return "This is the main page"


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    """
    Prediction Endpoint
    """
    prediction_file_path = request.get_json()["prediction_file_path"]
    dataframe = pd.read_csv(prediction_file_path)
    dataframe = dataframe.drop(["corporation", "exited"], axis=1)

    predictions = model_predictions(dataframe)
    return jsonify(predictions.tolist())


@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    """
    Scoring Endpoint, check the score of the deployed model
    """
    return score_model()


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    """
    Summary Statistics Endpoint
    """
    return jsonify(dataframe_summary())


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    """
    Diagnostics Endpoint
    """
    return jsonify(
        {
            "missing_percentage": missing_data(),
            "execution_time": execution_time(),
            "outdated_packages": outdated_packages_list(),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
