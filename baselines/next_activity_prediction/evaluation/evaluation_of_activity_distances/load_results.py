import pickle
from definitions import ROOT_DIR
import os
path_to_dict = os.path.join(ROOT_DIR, "evaluation", "evaluation_of_activity_distances", "intrinsic_evaluation",
                            "results.pkl")

results = pickle.load(open(path_to_dict, "rb"))

print(results)