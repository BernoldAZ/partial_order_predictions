import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import ast  # Safe parsing of dictionary-like strings
import json


def plot_heatmap(data_dict):
    # Convert dictionary values (ndarrays) into a 2D numpy array
    keys = list(data_dict.keys())  # Extract keys for labeling
    data_matrix = np.array([data_dict[key] for key in keys])  # Convert to 2D array

    # Create heatmap
    plt.figure(figsize=(12, 6))  # Set figure size
    sns.heatmap(data_matrix, cmap='coolwarm', annot=False, xticklabels=False, yticklabels=keys)

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Keys")
    plt.title("Heatmap of Dictionary Values")

    # Show plot
    plt.show()


