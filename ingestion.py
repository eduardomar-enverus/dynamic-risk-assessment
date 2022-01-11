import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob
from pathlib import Path


#############Load config.json and get input and output paths
from fullprocess import find_csv_files

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def write_files(final_df, ingested_files):
    if not os.path.exists(output_folder_path):
        os.makedirs(
            output_folder_path
        )  # If ingest data directory does not exist create it

    filename = os.path.join(output_folder_path, "finaldata.csv")
    final_df.to_csv(filename, index=False)

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as output:
        for file in ingested_files:
            output.write(f"{file}\n")


def merge_multiple_dataframe():
    ingested_files = find_csv_files(input_folder_path)
    final_df = pd.DataFrame()
    for filename in ingested_files:
        temp_df = pd.read_csv(os.path.join(input_folder_path,filename))
        final_df = pd.concat([final_df, temp_df])
    final_df = final_df.drop_duplicates().reset_index(
        drop=True
    )  # dedup combined dataframe
    write_files(final_df, ingested_files)


if __name__ == "__main__":
    final_df, ingested_files = merge_multiple_dataframe()
    write_files(final_df, ingested_files)
