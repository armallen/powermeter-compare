import argparse
import sys
from pathlib import Path

import fitparse
import matplotlib.pyplot as plt
import pandas as pd


# Define a function to read a FIT file and return a pandas DataFrame
def read_fit_file(file_path, power_coefficient=1.0):
    # Open the FIT file
    fitfile = fitparse.FitFile(file_path)
    # Create empty lists to hold the data
    power_data = []
    time_data = []

    # Iterate over all messages in the FIT file
    for record in fitfile.get_messages("record"):
        # Iterate over all data fields in the message
        power = None
        time = None

        for data in record:
            # If the data is power, add it to the power_data list
            if data.name == "power":
                power = data.value
            # If the data is time, add it to the time_data list
            elif data.name == "timestamp":
                time = data.value
        if power and time:
            power_data.append(power * power_coefficient)
            time_data.append(time)

    # Create a pandas DataFrame from the power and time data
    data = {"power": power_data, "time": time_data}
    print(len(power_data))
    print(len(time_data))
    df = pd.DataFrame(data)

    # Convert the time column to a datetime format
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return df


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process two file paths.")

    # Add the required file path arguments
    parser.add_argument("--fit-1", type=Path, help="Path to the first file.")
    parser.add_argument("--fit-2", type=Path, help="Path to the second file.")

    # Parse the arguments
    args = parser.parse_args()

    # Define the file paths for the two FIT files
    file_path_1 = sys.argv[1]
    file_path_2 = sys.argv[2]

    # Read the two FIT files into pandas DataFrames
    df_1 = read_fit_file(file_path_1)
    df_2 = read_fit_file(file_path_2)

    # Merge the two DataFrames on the time column
    merged_df = pd.merge(df_1, df_2, on="time", suffixes=("_1", "_2"))

    # Create a plot of the power values over time
    mean_1 = merged_df["power_1"].mean()
    mean_2 = merged_df["power_2"].mean()
    std_dev_1 = merged_df["power_1"].std()
    std_dev_2 = merged_df["power_2"].std()
    print(f"Mean 1: {mean_1}")
    print(f"Mean 2: {mean_2}")
    print(f"Std 1: {std_dev_1}")
    print(f"Std 2: {std_dev_2}")
    plt.plot(merged_df["time"], merged_df["power_1"], label="File 1")
    plt.plot(merged_df["time"], merged_df["power_2"], label="File 2")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    plt.show()
