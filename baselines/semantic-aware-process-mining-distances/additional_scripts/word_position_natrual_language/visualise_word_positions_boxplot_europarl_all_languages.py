import nltk
import os
import collections
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import europarl_raw
from matplotlib import rc
import numpy as np
from definitions import ROOT_DIR

# Set LaTeX-style fonts
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})

# Ensure necessary NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
#nltk.download('europarl_raw')

# Define supported languages
languages = [
    "danish", "dutch", "english", "finnish", "french", "german",
    "greek", "italian", "portuguese", "spanish", "swedish"
]

# Load the corpora for each language using europarl_raw
#corpus = {lang: getattr(europarl_raw, lang).raw() for lang in languages}



def read_europarl_dataset(lang):
    if lang == "danish":
        file_name = "europarl-v7.da-en.da"
    elif lang == "dutch":
        file_name = "europarl-v7.nl-en.nl"
    elif lang == "english":
        file_name = "europarl-v7.de-en.en"
    elif lang == "finnish":
        file_name = "europarl-v7.fi-en.fi"
    elif lang == "french":
        file_name = "europarl-v7.fr-en.fr"
    elif lang == "german":
        file_name = "europarl-v7.de-en.de"
    elif lang == "greek":
        file_name = "europarl-v7.el-en.el"
    elif lang == "italian":
        file_name = "europarl-v7.it-en.it"
    elif lang == "portuguese":
        file_name = "europarl-v7.pt-en.pt"
    elif lang == "spanish":
        file_name = "europarl-v7.es-en.es"
    elif lang == "swedish":
        file_name = "europarl-v7.sv-en.sv"

    file_path = os.path.join(ROOT_DIR, "additional_scripts", "word_position_natrual_language", "europarl_dataset", file_name)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


# Function to preprocess text
def preprocess_text(text):
    sentences = text.strip().split("\n")  # Split sentences using line breaks
    print(len(sentences))
    return [[word for word in word_tokenize(sentence.lower()) if word.isalnum()] for sentence in sentences]

# Function to compute relative word positions
def compute_relative_positions(traces, lang, with_stop_words):
    all_words = [word for trace in traces for word in trace]
    if with_stop_words:
        word_counts = collections.Counter(all_words)
    #only look at none stopwords
    else:
        stop_words = set(stopwords.words(lang))
        word_counts = collections.Counter(all_words)
        for word in list(word_counts.keys()):  # Iterate over a copy of the keys
            if word in stop_words:
                del word_counts[word]  # Remove stop words from Counter

    top_words = [word for word, _ in word_counts.most_common(20)]
    print(top_words)
    word_positions = {word: [] for word in top_words}

    for trace in traces:
        trace_length = len(trace)
        for idx, word in enumerate(trace):
            if word in top_words:
                relative_pos = idx / trace_length
                word_positions[word].append(relative_pos)

    return word_positions

# Process each language and prepare data for plotting
data = []
for lang in languages:
    print(lang)
    sentences = preprocess_text(read_europarl_dataset(lang))
    relative_positions = compute_relative_positions(sentences, lang, with_stop_words=True)

    # Data for all words
    data_all = [(lang, "All Words", word, pos) for word, positions in relative_positions.items() for pos in positions]

    # Filter stopwords
    filtered_positions = compute_relative_positions(sentences, lang, with_stop_words=False)
    data_filtered = [(lang, "Without Stopwords", word, pos) for word, positions in filtered_positions.items() for pos in positions]

    data.extend(data_all)
    data.extend(data_filtered)

df = pd.DataFrame(data, columns=["Language", "Type", "Word", "Relative Position"])

# Set up number of columns and rows for the plot grid
ncols = 6  # 4 columns
nrows = (len(languages) * 2 + ncols - 1) // ncols  # Two plots per language (all words + filtered)

#fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))  # Adjust figure size
fig, axes = plt.subplots(nrows, ncols, figsize=(26, 4 * nrows))  # Adjust figure size

axes = axes.flatten()  # Flatten axes array for easy iteration

# Keep track of which columns have plots in the last row
last_row_filled = [False] * ncols

for i, (ax, (lang_type, lang_data)) in enumerate(zip(axes, df.groupby(["Language", "Type"]))):
    row, col = divmod(i, ncols)  # Move this before using `col`

    top_words = lang_data["Word"].unique()[:5]  # Ensure only top 5 words are plotted
    lang_data = lang_data[lang_data["Word"].isin(top_words)]  # Filter dataset

    # Create boxplot
    sns.boxplot(
        x="Word",
        y="Relative Position",
        data=lang_data,
        ax=ax,
        whis=[0, 100],
        hue="Word",  # Assign x-variable to hue
        palette="Set2",
        width=0.6,
        legend=False  # Avoid duplicate legends
    )

    # Set title and labels
    lang, type_label = lang_type  # Extract language and type ("All Words" or "Without Stopwords")
    ax.set_title(f"{lang.capitalize()} - {type_label}", fontsize=20)

    # Y-axis label only for the first column
    if col == 0:
        ax.set_ylabel("Relative Word Positions", fontsize=20)
    else:
        ax.set_ylabel("")

    # X-axis label only for the last row
    if row == nrows - 1:
        ax.set_xlabel("Five most frequent activities", fontsize=20)
    else:
        ax.set_xlabel("")


    # Hide x-axis labels to avoid clutter
    ax.set_xticklabels([])

    # Track which columns have plots in the last row
    if row == nrows - 1:
        last_row_filled[col] = True

# Assign xlabel to the last available row in each column
for col in range(ncols):
    for row in reversed(range(nrows)):  # Start from the last row and move up
        idx = row * ncols + col
        if idx < len(axes) and axes[idx] in fig.axes:  # Check if valid axis
            axes[idx].set_xlabel("Five most frequent words", fontsize=20)
            break  # Assign only once per column

# Hide unused axes if the number of languages is not a multiple of ncols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("word_positions_all_languages_boxplots.pdf", format="pdf", transparent=True)
plt.show()
