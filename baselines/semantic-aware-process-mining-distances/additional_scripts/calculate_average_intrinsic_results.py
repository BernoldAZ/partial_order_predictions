import os
import pickle
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



EVENT_LOGS_DIR = os.path.join(ROOT_DIR , "evaluation", "evaluation_of_activity_distances", "event_logs", "intrinsic_evaluation", "results")

#all_logs = ["Sepsis"]

#all logs
#"""
all_logs = ['BPIC12',
            'BPIC12_A',
            'BPIC12_Complete',
            'BPIC12_O',
            'BPIC12_W',
            'BPIC12_W_Complete',
            'BPIC13_closed_problems',
            'BPIC13_incidents',
            'BPIC13_open_problems',
            'BPIC15_1',
            'BPIC15_2',
            'BPIC15_3',
            'BPIC15_4',
            'BPIC15_5',
            'BPIC17',
            'BPIC18',
            'BPIC19',
            'BPIC20_DomesticDeclarations',
            'BPIC20_InternationalDeclarations',
            'BPIC20_PermitLog',
            'BPIC20_PrepaidTravelCost',
            'BPIC20_RequestForPayment',
            'CCC19',
            'Env Permit',
            'Helpdesk',
            'Hospital Billing',
            'RTFM',
            'Sepsis']
#"""

#Logs where Autoencoder produced results
""" 
all_logs = [
            'BPIC12_O',
            'BPIC12_W',
            'BPIC13_closed_problems',
            'BPIC13_incidents',
            'BPIC13_open_problems',
            'BPIC15_1',
            'BPIC15_2',
            'BPIC15_3',
            'BPIC15_4',
            'BPIC15_5',
            'BPIC20_DomesticDeclarations',
            'BPIC20_InternationalDeclarations',
            'BPIC20_PermitLog',
            'BPIC20_PrepaidTravelCost',
            'BPIC20_RequestForPayment',
            'CCC19',
            'Env Permit',
            'Helpdesk',
            'Sepsis']
"""
log_statistics = {}

df_list = []

# Loop through each log directory
for log_name in all_logs:
    df_avg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg", log_name)

    if os.path.exists(df_avg_dir):  # Check if directory exists
        # Find the pickle file
        pickle_files = [f for f in os.listdir(df_avg_dir) if f.endswith(".pkl")]

        if pickle_files:  # If a pickle file exists
            pickle_path = os.path.join(df_avg_dir, pickle_files[0])  # Assuming only one pickle file per directory
            df = pd.read_pickle(pickle_path)  # Load DataFrame
            df_list.append(df)  # Append to the list

# Merge all DataFrames
if df_list:
    # Concatenate all DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    # Ensure 'Distance Function' is a string and sort alphabetically
    combined_df['Distance Function'] = combined_df['Distance Function'].astype(str)
    combined_df = combined_df.sort_values(by='Distance Function')

    # Compute the average per Distance Function
    avg_df = combined_df.groupby('Distance Function')[['diameter', 'precision@w-1', 'precision@1', 'triplet']].mean()

    # Ensure bar plot categories are sorted
    avg_df = avg_df.sort_index()

    # Create bar plots for averages
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    metrics = ['diameter', 'precision@w-1', 'precision@1', 'triplet']
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


    def get_metric_value(df, func, col):
        """Return the metric rounded to 2 decimals (or blank if not available)."""
        try:
            val = df.loc[func, col]
            return f"{val:.2f}"
        except KeyError:
            return ""

    # --- Assumptions ---
    # 1. You already computed the DataFrame 'avg_df' where:
    #      - The index is the distance function name.
    #      - Columns: 'diameter', 'precision@w-1', 'precision@1', 'triplet'.
    # 2. The list 'table_entries' maps each row in your table to the corresponding distance function.
    # 3. The LaTeX package 'xcolor' (or similar) is loaded in your LaTeX preamble to support \textcolor.

    # --- Define the table entries ---
    # Each tuple: (Method, Context, PMI, Window Size, distance_function)
    table_entries = [
        # --- Our New Methods: Activity-Activity Co-Occurrence Matrix ---
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "No", 3,
         "Activity-Activitiy Co Occurrence Bag Of Words w_3"),
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "No", 5,
         "Activity-Activitiy Co Occurrence Bag Of Words w_5"),
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "No", 9,
         "Activity-Activitiy Co Occurrence Bag Of Words w_9"),

        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "PMI", 3,
         "Activity-Activitiy Co Occurrence Bag Of Words PMI w_3"),
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "PMI", 5,
         "Activity-Activitiy Co Occurrence Bag Of Words PMI w_5"),
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "PMI", 9,
         "Activity-Activitiy Co Occurrence Bag Of Words PMI w_9"),

        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "PPMI", 3,
         "Activity-Activitiy Co Occurrence Bag Of Words PPMI w_3"),
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "PPMI", 5,
         "Activity-Activitiy Co Occurrence Bag Of Words PPMI w_5"),
        ("Activity-Activity Co-Occurrence Matrix", "Multiset", "PPMI", 9,
         "Activity-Activitiy Co Occurrence Bag Of Words PPMI w_9"),

        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "No", 3,
         "Activity-Activitiy Co Occurrence N-Gram w_3"),
        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "No", 5,
         "Activity-Activitiy Co Occurrence N-Gram w_5"),
        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "No", 9,
         "Activity-Activitiy Co Occurrence N-Gram w_9"),

        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "PMI", 3,
         "Activity-Activitiy Co Occurrence N-Gram PMI w_3"),
        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "PMI", 5,
         "Activity-Activitiy Co Occurrence N-Gram PMI w_5"),
        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "PMI", 9,
         "Activity-Activitiy Co Occurrence N-Gram PMI w_9"),

        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "PPMI", 3,
         "Activity-Activitiy Co Occurrence N-Gram PPMI w_3"),
        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "PPMI", 5,
         "Activity-Activitiy Co Occurrence N-Gram PPMI w_5"),
        ("Activity-Activity Co-Occurrence Matrix", "Sequence", "PPMI", 9,
         "Activity-Activitiy Co Occurrence N-Gram PPMI w_9"),

        # --- Our New Methods: Activity-Context Frequency Matrix ---
        ("Activity-Context Frequency Matrix", "Multiset", "No", 3, "Activity-Context Bag Of Words w_3"),
        ("Activity-Context Frequency Matrix", "Multiset", "No", 5, "Activity-Context Bag Of Words w_5"),
        ("Activity-Context Frequency Matrix", "Multiset", "No", 9, "Activity-Context Bag Of Words w_9"),

        ("Activity-Context Frequency Matrix", "Multiset", "PMI", 3, "Activity-Context Bag Of Words PMI w_3"),
        ("Activity-Context Frequency Matrix", "Multiset", "PMI", 5, "Activity-Context Bag Of Words PMI w_5"),
        ("Activity-Context Frequency Matrix", "Multiset", "PMI", 9, "Activity-Context Bag Of Words PMI w_9"),

        ("Activity-Context Frequency Matrix", "Multiset", "PPMI", 3, "Activity-Context Bag Of Words PPMI w_3"),
        ("Activity-Context Frequency Matrix", "Multiset", "PPMI", 5, "Activity-Context Bag Of Words PPMI w_5"),
        ("Activity-Context Frequency Matrix", "Multiset", "PPMI", 9, "Activity-Context Bag Of Words PPMI w_9"),

        ("Activity-Context Frequency Matrix", "Sequence", "No", 3, "Activity-Context N-Grams w_3"),
        ("Activity-Context Frequency Matrix", "Sequence", "No", 5, "Activity-Context N-Grams w_5"),
        ("Activity-Context Frequency Matrix", "Sequence", "No", 9, "Activity-Context N-Grams w_9"),

        ("Activity-Context Frequency Matrix", "Sequence", "PMI", 3, "Activity-Context N-Grams PMI w_3"),
        ("Activity-Context Frequency Matrix", "Sequence", "PMI", 5, "Activity-Context N-Grams PMI w_5"),
        ("Activity-Context Frequency Matrix", "Sequence", "PMI", 9, "Activity-Context N-Grams PMI w_9"),

        ("Activity-Context Frequency Matrix", "Sequence", "PPMI", 3, "Activity-Context N-Grams PPMI w_3"),
        ("Activity-Context Frequency Matrix", "Sequence", "PPMI", 5, "Activity-Context N-Grams PPMI w_5"),
        ("Activity-Context Frequency Matrix", "Sequence", "PPMI", 9, "Activity-Context N-Grams PPMI w_9"),

        # --- Existing Methods ---
        ("Unit Distance", "-", "-", "-", "Unit Distance"),

        ("Bose Substitution Scores", "Sequence", "-", "3", "Bose 2009 Substitution Scores w_3"),
        ("Bose Substitution Scores", "Sequence", "-", "5", "Bose 2009 Substitution Scores w_5"),
        ("Bose Substitution Scores", "Sequence", "-", "9", "Bose 2009 Substitution Scores w_9"),

        ("act2vec (Multiset)", "Multiset", "-", "3", "De Koninck 2018 act2vec CBOW w_3"),
        ("act2vec (Multiset)", "Multiset", "-", "5", "De Koninck 2018 act2vec CBOW w_5"),
        ("act2vec (Multiset)", "Multiset", "-", "9", "De Koninck 2018 act2vec CBOW w_9"),

        ("act2vec (Sequence)", "Sequence", "-", "3", "De Koninck 2018 act2vec skip-gram w_3"),
        ("act2vec (Sequence)", "Sequence", "-", "5", "De Koninck 2018 act2vec skip-gram w_5"),
        ("act2vec (Sequence)", "Sequence", "-", "9", "De Koninck 2018 act2vec skip-gram w_9"),

        ("Embedding Process Structure", "Process Model", "-", "-", "Chiorrini 2022 Embedding Process Structure")
    ]

    # --- Ranking and Highlighting ---

    # Helper function to safely get numeric values from avg_df
    def get_metric_numeric(df, func, col):
        try:
            return df.loc[func, col]
        except KeyError:
            return None

    # Create a highlight mapping dictionary.
    # For each metric column we store a dictionary mapping the row index to a highlight color.
    # The keys are the column names as used in avg_df.
    highlight_map = {"diameter": {}, "precision@1": {}, "precision@w-1": {}, "triplet": {}}
    # Specify the sort order per metric: for diameter 'asc' (lower is better) and for the others 'desc'
    metric_info = {
        "diameter": {"order": "asc"},
        "precision@1": {"order": "desc"},
        "precision@w-1": {"order": "desc"},
        "triplet": {"order": "desc"}
    }

    # For each metric, rank the table entries and mark the top three.
    for col, info in metric_info.items():
        order = info["order"]
        values = []
        for i, entry in enumerate(table_entries):
            # entry: (Method, Context, PMI, Window, distance_function)
            dist_func = entry[4]
            val = get_metric_numeric(avg_df, dist_func, col)
            if val is not None:
                values.append((i, val))
        # Sort by the appropriate order.
        if order == "asc":
            sorted_vals = sorted(values, key=lambda x: x[1])
        else:
            sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)
        # Mark the top three (if available)
        for rank, (i, v) in enumerate(sorted_vals[:3]):
            if rank == 0:
                color = "red"
            elif rank == 1:
                color = "blue"
            elif rank == 2:
                color = "green"
            highlight_map[col][i] = color

    # Function to format a metric value: rounds to 2 decimals and applies color highlighting if needed.
    def format_metric(df, func, col, row_index, highlight_map):
        try:
            val = df.loc[func, col]
            formatted = f"{val:.2f}"
            if row_index in highlight_map.get(col, {}):
                color = highlight_map[col][row_index]
                formatted = f"\\textcolor{{{color}}}{{{formatted}}}"
            return formatted
        except KeyError:
            return ""

    # --- Generate the LaTeX Table ---

    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\vspace*{-1.9cm}")  # adjust vertical spacing as needed
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
    latex_lines.append(r"\hline")
    latex_lines.append(
        r"\parbox[t]{3mm}{\multirow{36}{*}{\rotatebox[origin=c]{90}{Our New Methods}}} & \textbf{Method} & \textbf{Context} & \textbf{PMI} & \textbf{\makecell{Window\\Size}} & $I_{diameter}$ & $I_{nn}$ & $I_{prec}$ & $I_{triplet}$ \\")
    latex_lines.append(r"\cline{1-9}")

    # Loop through each table entry and print a row.
    for i, entry in enumerate(table_entries):
        method, context, pmi, window, dist_func = entry
        # Format each metric value using our helper (with highlighting if applicable)
        diam = format_metric(avg_df, dist_func, "diameter", i, highlight_map)
        prec1 = format_metric(avg_df, dist_func, "precision@1", i, highlight_map)
        prec_w = format_metric(avg_df, dist_func, "precision@w-1", i, highlight_map)
        trip = format_metric(avg_df, dist_func, "triplet", i, highlight_map)

        window_str = str(window) if window not in [None, "-"] else "-"
        pmi_str = pmi if pmi not in [None, ""] else "-"

        # Construct the LaTeX row.
        row = f" & {method} & {context} & {pmi_str} & {window_str} & {diam} & {prec1} & {prec_w} & {trip} \\\\"
        latex_lines.append(row)
        # (You can add \cline commands here to mimic your original table layout if desired.)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\captionsetup{skip=4pt}")
    latex_lines.append(r"\caption{Recreated Table}")
    latex_lines.append(r"\label{tab:recreated_table}")
    latex_lines.append(r"\end{table}")

    latex_table = "\n".join(latex_lines)
    print(latex_table)
else:
    print("No valid dataframes found in the specified directories.")
