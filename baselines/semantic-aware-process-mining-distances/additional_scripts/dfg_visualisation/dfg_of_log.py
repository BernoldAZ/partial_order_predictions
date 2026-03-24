import nltk
from nltk.corpus import gutenberg, europarl_raw, udhr
from nltk.tokenize import TweetTokenizer, sent_tokenize
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from matplotlib import rc
from itertools import cycle  # For cycling markers per category
import os

# Define event log directory
EVENT_LOGS_DIR = os.path.join(ROOT_DIR, "event_logs")



event_log = pm4py.read_xes(EVENT_LOGS_DIR + "/BPIC13_closed_problems.xes.gz")

# Discover the workflow net
dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
print("dfg discovered")
pm4py.view_dfg(dfg, start_activities, end_activities,format='pdf')
#pm4py.objects.petri_net.exporter.variants.pnml.export_net(net, im, "austen-emma-net", final_marking=fm, export_prom5=False, parameters=None)
# Print example output
#print(tokenized_sentences[:5])  # Show first 5 cleaned sentences


event_log = pm4py.read_xes(EVENT_LOGS_DIR + "/BPIC17.xes.gz")

# Discover the workflow net
dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
print("dfg discovered")
pm4py.view_dfg(dfg, start_activities, end_activities,format='pdf')
