import os
import pickle
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

EVENT_LOGS_DIR = os.path.join(ROOT_DIR , "evaluation", "evaluation_of_activity_distances", "next_activity_prediction", "results_everman")

all_logs = [
            'BPIC12_A',
            'BPIC12_O',
            'BPIC12_W',
            'BPIC13_closed_problems',
            'BPIC13_incidents',
            'BPIC13_open_problems',
            'BPIC15_1',
            'BPIC15_2',
            'BPIC15_3',
            'BPIC15_4',
            "BPIC15_5",
            'BPI_2020_DomesticDeclarations',
            'BPI_2020_InternationalDeclarations',
            'BPI_2020_PermitLog',
            'BPI_2020_PrepaidTravelCost',
            'BPI_2020_RequestForPayment',
            'SEPSIS',
    "Helpdesk",
    "env_permit",
    "nasa"
]




#all_logs = ["nasa"]
log_statistics = {}


df_list = []

# Loop through each log directory
# Loop through each log directory
for log_name in all_logs:
    df_avg_dir = os.path.join(ROOT_DIR, EVENT_LOGS_DIR, log_name)

    if os.path.exists(df_avg_dir):
        for item in os.listdir(df_avg_dir):
            full_path = os.path.join(df_avg_dir, item)
            # Check if the item is a directory
            if os.path.isdir(full_path):
                with open(os.path.join(full_path,log_name+".txt"), "r") as file:
                    text = file.read()
                # Use regex to extract the numbers
                acc_match = re.search(r"Accuracy:\s*([0-9.]+)", text)
                f1_match = re.search(r"Weighted f1:\s*([0-9.]+)", text)
                if acc_match and f1_match:
                    accuracy = float(acc_match.group(1))
                    weighted_f1 = float(f1_match.group(1))
                    # Save log name, distance function (i.e. item), accuracy, and weighted f1 score
                    df_list.append({
                        "log_name": log_name,
                        "Distance Function": item,
                        "accuracy": accuracy,
                        "weighted_f1": weighted_f1
                    })

# Merge all DataFrames
if df_list:
    # Concatenate all DataFrames
    combined_df = pd.DataFrame(df_list)
    # Ensure 'Distance Function' is a string and sort alphabetically
    combined_df['Distance Function'] = combined_df['Distance Function'].astype(str)
    combined_df = combined_df.sort_values(by='Distance Function')

    # Compute the average per Distance Function
    avg_df = combined_df.groupby('Distance Function')[['accuracy', 'weighted_f1']].mean()

    # Ensure bar plot categories are sorted
    avg_df = avg_df.sort_index()

    # Create bar plots for averages
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    metrics = ['accuracy']
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        avg_df[metric].plot(kind='bar', ax=axes[i], title=f'Average {metric} by Distance Function', color='skyblue')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)  # Set y-axis limit from 0 to 1
        axes[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)  # Add dashed grid lines on y-axis

    plt.tight_layout()
    plt.show()

    # Create boxplots for distributions per Distance Function
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    # Get the sorted order of Distance Functions
    sorted_distance_functions = sorted(combined_df['Distance Function'].unique())

    for i, metric in enumerate(metrics):
        sns.boxplot(data=combined_df, x='Distance Function', y=metric, ax=axes[i], order=sorted_distance_functions)
        axes[i].set_title(f'Distribution of {metric} by Distance Function')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("Distance Function")
        axes[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels
        axes[i].set_ylim(0, 1)  # Set y-axis limit from 0 to 1
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)  # Add dashed grid lines on y-axis

    plt.tight_layout()
    plt.show()
    rounded_df = avg_df.round(2)
    print(rounded_df)

else:
    print("No valid dataframes found in the specified directories.")
