import joblib
import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
production_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions():
    # Load trained model
    model = joblib.load(os.path.join(production_deployment_path, 'trainedmodel.pkl'))

    # Load test data
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_true = test_data.pop('exited')
    test_data = test_data.drop(columns=['corporation'])
    # Predictions
    predictions = model.predict(test_data)
    return predictions

##################Function to get summary statistics
def dataframe_summary():
    data = pd.read_csv(os.path.join(dataset_csv_path,'finaldata.csv'))
    data = data.drop(columns=['exited','corporation'])

    summary_stats = data.agg(
        {
            "lastmonth_activity": ["min", "max", "mean", "median", "std"],
            "lastyear_activity": ["min", "max", "mean", "median", "std"],
            "number_of_employees": ["min", "max", "mean", "median", "std"],
        }
    )

    return summary_stats

##################Function to get timings
def execution_time():`
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    pass

if __name__ == '__main__':
    # model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
