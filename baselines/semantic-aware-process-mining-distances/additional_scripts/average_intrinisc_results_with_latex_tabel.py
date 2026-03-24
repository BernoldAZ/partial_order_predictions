import os
import pickle
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "event_logs", "intrinsic_evaluation", "results")

all_logs = ['BPIC12', 'BPIC12_A', 'BPIC12_Complete', 'BPIC12_O', 'BPIC12_W', 'BPIC12_W_Complete',
            'BPIC13_closed_problems', 'BPIC13_incidents', 'BPIC13_open_problems', 'BPIC15_1', 'BPIC15_2',
            'BPIC15_3', 'BPIC15_4', 'BPIC15_5', 'BPIC17', 'BPIC18', 'BPIC19', 'BPIC20_DomesticDeclarations',
            'BPIC20_InternationalDeclarations', 'BPIC20_PermitLog', 'BPIC20_PrepaidTravelCost',
            'BPIC20_RequestForPayment', 'CCC19', 'Env Permit', 'Helpdesk', 'Hospital Billing', 'RTFM', 'Sepsis']

df_list = []

# Load DataFrames
for log_name in all_logs:
    df_avg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg", log_name)

    if os.path.exists(df_avg_dir):
        pickle_files = [f for f in os.listdir(df_avg_dir) if f.endswith(".pkl")]

        if pickle_files:
            pickle_path = os.path.join(df_avg_dir, pickle_files[0])
            df = pd.read_pickle(pickle_path)
            df_list.append(df)

# Combine DataFrames
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['Distance Function'] = combined_df['Distance Function'].astype(str)
    combined_df = combined_df.sort_values(by='Distance Function')

    avg_df = combined_df.groupby('Distance Function')[['diameter', 'precision@w-1', 'precision@1', 'triplet']].mean()
    avg_df = avg_df.sort_index().round(2)

    # Print formatted results
    print("\nFormatted Average Results:")
    for idx, row in avg_df.iterrows():
        method, context, pmi, window_size = idx.split('_')
        print(f"Method: {method}, Context: {context}, PMI: {pmi}, Window Size: {window_size}, Diameter: {row['diameter']}, Precision@w-1: {row['precision@w-1']}, Precision@1: {row['precision@1']}, Triplet: {row['triplet']}")

    # Save results to LaTeX
    latex_file_path = os.path.join(ROOT_DIR, 'results_table.tex')
    with open(latex_file_path, 'w') as latex_file:
        prev_method_context_pmi = ("", "", "")
        for idx, row in avg_df.iterrows():
            method, context, pmi, window_size = idx.split('_')
            curr_method_context_pmi = (method, context, pmi)

            if curr_method_context_pmi != prev_method_context_pmi:
                latex_line = f"\\multirow{{3}}{{*}}{{{method}}} & \\multirow{{3}}{{*}}{{{context}}} & \\multirow{{3}}{{*}}{{{pmi}}} & {window_size} & {row['diameter']} & {row['precision@w-1']} & {row['precision@1']} & {row['triplet']} \\\\n"
                prev_method_context_pmi = curr_method_context_pmi
            else:
                latex_line = f" & & & {window_size} & {row['diameter']} & {row['precision@w-1']} & {row['precision@1']} & {row['triplet']} \\\\n"

            latex_file.write(latex_line)

    # Bar plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    metrics = ['diameter', 'precision@w-1', 'precision@1', 'triplet']
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        avg_df[metric].plot(kind='bar', ax=axes[i], title=f'Average {metric} by Distance Function', color='skyblue')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    sorted_distance_functions = sorted(combined_df['Distance Function'].unique())

    for i, metric in enumerate(metrics):
        sns.boxplot(data=combined_df, x='Distance Function', y=metric, ax=axes[i], order=sorted_distance_functions)
        axes[i].set_title(f'Distribution of {metric} by Distance Function')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("Distance Function")
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

else:
    print("No valid dataframes found in the specified directories.")
