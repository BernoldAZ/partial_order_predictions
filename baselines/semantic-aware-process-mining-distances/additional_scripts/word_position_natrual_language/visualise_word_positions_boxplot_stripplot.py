import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import TweetTokenizer, sent_tokenize
import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure necessary NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("gutenberg")

# Load corpus
sentences = gutenberg.sents("austen-emma.txt")


# Tokenization
def preprocess_sentences(sentences):
    tokenizer = TweetTokenizer()
    tokenized_sentences = [
        [word for word in tokenizer.tokenize(" ".join(sentence)) if word.isalnum()]
        for sentence in sentences if sentence
    ]
    return tokenized_sentences


tokenized_sentences = preprocess_sentences(sentences)


# Compute relative positions
def compute_relative_positions(traces):
    all_words = [word for trace in traces for word in trace]
    word_counts = collections.Counter(all_words)

    top_words = [word for word, _ in word_counts.most_common(20)]
    word_positions = {word: [] for word in top_words}

    for trace in traces:
        trace_length = len(trace)
        for idx, word in enumerate(trace):
            if word in top_words:
                relative_pos = idx / trace_length
                word_positions[word].append(relative_pos)

    return word_positions


# Compute relative positions for all words
log_statistics = compute_relative_positions(tokenized_sentences)

# Convert to DataFrame
data = []
for word, positions in log_statistics.items():
    for pos in positions:
        data.append([word, pos])
df_all = pd.DataFrame(data, columns=["Word", "Relative Position"])

# **Filter stop words and recompute relative positions**
stop_words = set(stopwords.words('english'))
filtered_sentences = [
    [word for word in sentence if word.lower() not in stop_words]
    for sentence in tokenized_sentences
]

# Compute relative positions for filtered words
log_statistics_filtered = compute_relative_positions(filtered_sentences)

# Convert to DataFrame
data_filtered = []
for word, positions in log_statistics_filtered.items():
    for pos in positions:
        data_filtered.append([word, pos])
df_filtered = pd.DataFrame(data_filtered, columns=["Word", "Relative Position"])

# **Plot both boxplots (STACKED but SEPARATE X-AXES)**
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Boxplot for all words (Separate X-axis)
sns.boxplot(
    data=df_all, x="Word", y="Relative Position",
    showcaps=True, width=0.95, fliersize=0,
    boxprops={'facecolor': 'none', 'edgecolor': 'black'},
    whiskerprops={'color': 'black'}, capprops={'color': 'black'},
    medianprops={'color': 'red'}, ax=axes[0]
)
sns.stripplot(
    data=df_all, x="Word", y="Relative Position",
    color="green", alpha=0.3, jitter=0.35, size=2, ax=axes[0]
)
axes[0].set_title("Relative Word Positions (All Words)")
axes[0].set_ylabel("Relative Position")
axes[0].grid(True)
axes[0].set_xticklabels(df_all["Word"].unique(), rotation=45)  # Ensure separate x labels

# Boxplot for filtered words (Separate X-axis)
sns.boxplot(
    data=df_filtered, x="Word", y="Relative Position",
    showcaps=True, width=0.95, fliersize=0,
    boxprops={'facecolor': 'none', 'edgecolor': 'black'},
    whiskerprops={'color': 'black'}, capprops={'color': 'black'},
    medianprops={'color': 'red'}, ax=axes[1]
)
sns.stripplot(
    data=df_filtered, x="Word", y="Relative Position",
    color="blue", alpha=0.3, jitter=0.35, size=2, ax=axes[1]
)
axes[1].set_title("Relative Word Positions (Filtered Stopwords)")
axes[1].set_xlabel("Top Words")
axes[1].set_ylabel("Relative Position")
axes[1].grid(True)
axes[1].set_xticklabels(df_filtered["Word"].unique(), rotation=45)  # Separate X labels

# Adjust layout
plt.tight_layout()
plt.show()
