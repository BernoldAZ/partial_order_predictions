import nltk
from nltk.corpus import gutenberg, europarl_raw, udhr
from nltk.tokenize import TweetTokenizer, sent_tokenize
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

# Ensure necessary NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("gutenberg")
nltk.download("europarl_raw")
nltk.download("udhr")


def get_udhr_sentences(lang_code):
    """Returns UDHR sentences if available, otherwise None"""
    try:
        return udhr.sents(lang_code)
    except OSError:
        print(f"Warning: UDHR corpus for {lang_code} not found. Skipping...")
        return None


def get_europarl_sentences(language_corpus):
    """Returns sentences for the Europarl corpus by manually splitting text"""
    try:
        raw_text = " ".join(language_corpus.words())  # Get full text
        sentences = [sent.split() for sent in sent_tokenize(raw_text)]  # Split into sentences
        return sentences
    except AttributeError:
        print(f"Warning: Europarl corpus for {language_corpus} not found.")
        return None


# Define corpus sources
LANGUAGES = {
    "English": gutenberg.sents("austen-emma.txt"),
    "Spanish": get_europarl_sentences(europarl_raw.spanish),
    "French": get_europarl_sentences(europarl_raw.french),
    "German": get_europarl_sentences(europarl_raw.german),
    "Italian": get_europarl_sentences(europarl_raw.italian),
    "Dutch": get_europarl_sentences(europarl_raw.dutch),
    "Chinese": get_udhr_sentences("Chinese-UTF8"),  # Try another available option
    "Japanese": get_udhr_sentences("ja"),
    "Korean": get_udhr_sentences("ko"),
    "Persian": get_udhr_sentences("fa"),
}


# Function to preprocess sentences
def preprocess_sentences(sentences):
    """
    Tokenizes and cleans sentences, removing punctuation while keeping sentence structure.

    :param sentences: List of tokenized raw sentences.
    :return: List of tokenized sentences without punctuation.
    """
    tokenizer = TweetTokenizer()

    tokenized_sentences = [
        [word for word in tokenizer.tokenize(" ".join(sentence)) if word.isalnum()]
        for sentence in sentences if sentence  # Ensure sentence is not empty
    ]

    return tokenized_sentences


# Example: Process English sentences
tokenized_sentences = preprocess_sentences(LANGUAGES["English"])

event_log = EventLog()

# Transform the list of traces into an EventLog object
for trace_id, trace in enumerate(tokenized_sentences[:100]):
    pm4py_trace = Trace()
    for event_id, activity in enumerate(trace):
        # Create an event with attributes
        event = Event({
            DEFAULT_NAME_KEY: activity,  # 'concept:name' for activity name
            "trace_id": trace_id,  # Custom trace attribute
            "event_index": event_id  # Index of the event in the trace
        })
        pm4py_trace.append(event)  # Add event to the trace
    event_log.append(pm4py_trace)  # Add trace to the event log

# Discover the workflow net
dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
print("dfg discovered")
pm4py.view_dfg(dfg, start_activities, end_activities,format='pdf')
#pm4py.objects.petri_net.exporter.variants.pnml.export_net(net, im, "austen-emma-net", final_marking=fm, export_prom5=False, parameters=None)
# Print example output
#print(tokenized_sentences[:5])  # Show first 5 cleaned sentences
