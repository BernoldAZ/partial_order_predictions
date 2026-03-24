import pandas as pd
import os
from definitions import ROOT_DIR

# Path to the CSV file containing the runtime results.
csv_file = 'runtime_results_10_repetitions.csv'
df = pd.read_csv(os.path.join(ROOT_DIR, "results", "activity_distances", csv_file))

# Round the 'average_duration' column to three decimal places.
df['average_duration'] = df['average_duration'].round(3)

# Define the order of logs (modify this to your preferred order)
logs_order = sorted(df['log'].unique())

# Print the header row with log names (the first cell indicates the parameter).
header_row = "Activity Function Name & " + " & ".join(logs_order) + r" \\"
print(header_row)
print(r"\hline")

# Group the DataFrame by activity_function_name.
grouped = df.groupby('activity_function_name')

# For each activity_function_name, print one LaTeX-formatted row.
for func_name, group in grouped:
    # For every log in the ordered list, get the average_duration.
    # If an entry for a log is missing, leave that cell empty.
    values = []
    for log in logs_order:
        sub = group[group['log'] == log]
        if not sub.empty:
            # Assumes exactly one entry per function-log pair.
            values.append(str(sub.iloc[0]['average_duration']))
        else:
            values.append("")  # Leave empty if there is no value.
    latex_row = f"{func_name} & " + " & ".join(values) + r" \\"
    print(latex_row)
