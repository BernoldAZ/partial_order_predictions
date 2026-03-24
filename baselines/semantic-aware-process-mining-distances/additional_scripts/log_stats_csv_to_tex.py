#!/usr/bin/env python3
import os
import sys
import pandas as pd
from definitions import ROOT_DIR


def generate_latex_table():
    csv_path = os.path.join(ROOT_DIR, "log_stats.csv")
    # First try to read the CSV using automatic delimiter detection.
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Remove extra whitespace from column names.
    df.columns = df.columns.str.strip()

    # If the DataFrame is loaded as one column and the header contains commas,
    # then assume the file is comma-separated.
    if len(df.columns) == 1 and ',' in df.columns[0]:
        df = pd.read_csv(csv_path, delimiter=",")
        df.columns = df.columns.str.strip()

    # Define the expected column names.
    expected_columns = [
        "log_name",
        "unique_activities",
        "num_traces",
        "ratio_trace_variants",
        "min_trace_length",
        "avg_trace_length",
        "max_trace_length"
    ]
    # Check if any expected columns are missing.
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print(f"Error: The following expected columns are missing in the CSV file: {missing}")
        sys.exit(1)

    # Define columns that are numeric and need to be rounded.
    float_columns = [
        "unique_activities",
        "num_traces",
        "ratio_trace_variants",
        "min_trace_length",
        "avg_trace_length",
        "max_trace_length"
    ]

    # Helper function to parse numeric strings.
    def parse_numeric(val):
        s = str(val)
        # Remove extra dots if there are multiple (e.g. due to thousand separators)
        if s.count(".") > 1:
            s = s.replace(".", "")
        try:
            return float(s)
        except Exception:
            return s  # leave unchanged if conversion fails

    # Process each numeric column: parse and then format the value.
    for col in float_columns:
        df[col] = df[col].apply(parse_numeric)

        def format_value(x):
            if isinstance(x, float):
                # If the float is effectively an integer, display without decimals.
                if x.is_integer():
                    return f"{int(x)}"
                else:
                    return f"{x:.2f}"
            return x

        df[col] = df[col].apply(format_value)

    # LaTeX header for the table.
    header = r"""\begin{table}[t]
	\caption{Descriptive statistics of the evaluated event logs.}
	\vspace{0.5em}
	\resizebox{\textwidth}{!}{%
		\begin{tabular}{@{}lccccccccc@{}}
			\toprule
			&       &       &        & \multicolumn{3}{c}{\textbf{Trace Length}} & \multicolumn{3}{c}{\textbf{Used for}}   \\ \cmidrule(lr){5-7} \cmidrule(lr){8-10} 
			\multirow{-2}{*}{\textbf{Event Log}} &
			\multirow{-2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}\# Unique\\Activities\end{tabular}}} &
			\multirow{-2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}\# of \\ Traces\end{tabular}}} &
			\multirow{-2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}$\frac{\# of Variants}{\# of Traces}$\end{tabular}}} &
			\multicolumn{1}{l}{\textbf{Min.}} &
			\multicolumn{1}{l}{\textbf{Avg.}} &
			\multicolumn{1}{l}{\textbf{Max.}} &    
			\multicolumn{1}{l}{\textbf{Intrinsic}} &
			\multicolumn{1}{l}{\textbf{N. Act.}} & \multicolumn{1}{l}{\textbf{Runtime}}\\ \midrule
"""
    footer = r"""			\bottomrule
		\end{tabular}%
	}
	\label{tab:logstatistics}
	\vspace{-1em}
\end{table}"""

    # Construct table rows from the DataFrame.
    # The columns are assumed to be:
    #   log_name, unique_activities, num_traces, ratio_trace_variants,
    #   min_trace_length, avg_trace_length, and max_trace_length.
    # The last three columns ("Used for") are left empty.
    rows = ""
    for _, row in df.iterrows():
        row_tex = (
            f"    {row['log_name']} & "
            f"{row['unique_activities']} & "
            f"{row['num_traces']} & "
            f"{row['ratio_trace_variants']} & "
            f"{row['min_trace_length']} & "
            f"{row['avg_trace_length']} & "
            f"{row['max_trace_length']} & "
            " &  &  \\\\ \n"
        )
        rows += row_tex

    latex_code = header + "\n" + rows + footer
    return latex_code


if __name__ == "__main__":
    latex_table = generate_latex_table()
    print(latex_table)
