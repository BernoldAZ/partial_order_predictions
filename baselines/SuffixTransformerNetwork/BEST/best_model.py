"""BEST: Bilaterally Expanding Subtrace Tree for Event Sequence Prediction.

Based on:
    Rauch, S., Frey, C. M. M., Maldonado, A. J., & Seidl, T. (2025).
    BEST: Bilaterally Expanding Subtrace Tree for Event Sequence Prediction.
    In Business Process Management (BPM 2025). Springer, LNCS 16044.
    https://link.springer.com/chapter/10.1007/978-3-032-02867-9_25

BEST is a non-parametric, pattern-mining-based baseline for activity suffix
prediction in Predictive Process Monitoring (PPM). It builds a hierarchical
n-gram lookup structure from training traces, then generates activity suffixes
by greedily selecting the most probable next activity given the longest
matching context found in the structure.

Key properties:
    - Control-flow only (NDA): uses only activity labels, no timestamp or
      case/event attributes.
    - No gradient-based training: the "fit" step counts pattern frequencies
      in training data (analogous to building a language model n-gram table).
    - Suffix generation is autoregressive and greedy (argmax decoding).
    - Falls back to shorter contexts (and ultimately to the unigram mode) when
      no matching n-gram context is found.
"""

from collections import defaultdict
import torch
from tqdm import tqdm


def _int_defaultdict():
    """Module-level factory for defaultdict(int).

    A named top-level callable is used instead of a lambda so that
    BESTModel instances are picklable (lambdas cannot be pickled).
    """
    return defaultdict(int)


class BESTModel:
    """Bilaterally Expanding Subtrace Tree (BEST) model for activity suffix
    prediction.

    The model stores conditional next-activity counts for all observed
    context subsequences up to `max_context_length` in the training data.
    During prediction it performs longest-context-first lookup (back-off
    n-gram style) and returns the argmax activity.

    Parameters
    ----------
    num_activities : int
        Total number of distinct activity classes including the padding
        token (index 0) and the END token (index ``num_activities - 1``).
    max_context_length : int, optional
        Maximum number of preceding activities used as context when
        looking up the conditional distribution.  Longer histories are
        truncated to this length during both fitting and prediction.
        Smaller values reduce memory usage and fitting time at the cost
        of weaker context sensitivity.  By default 10.
    """

    def __init__(self, num_activities, max_context_length=10):
        self.num_activities = num_activities
        self.end_token = num_activities - 1
        self.max_context_length = max_context_length

        # ngram_counts[context_tuple][next_activity_int] -> int count
        self.ngram_counts = defaultdict(_int_defaultdict)
        # unigram_counts[activity_int] -> int count  (0-order fallback)
        self.unigram_counts = defaultdict(int)

        self.fitted = False
        self._unigram_most_common = self.end_token  # safe default before fit

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, train_dataset, num_categoricals_pref):
        """Build the BEST pattern tree from the training dataset.

        For every training instance the complete trace is reconstructed as
        the concatenation of the (non-padded) prefix activities and the
        suffix label activities up to and including the END token.  All
        sub-sequences of that trace up to ``max_context_length`` are then
        added to the n-gram count table.

        Parameters
        ----------
        train_dataset : tuple of torch.Tensor
            Training dataset in the format produced by the preprocessing
            pipeline (``Preprocessing/from_log_to_tensors.py``).

            Relevant tensors:

            * ``train_dataset[num_categoricals_pref - 1]`` --
              integer-encoded prefix activity labels,
              shape ``(N, W)``, right-padded with zeros, dtype int64.
            * ``train_dataset[num_categoricals_pref + 1]`` --
              boolean padding mask, shape ``(N, W)``,
              ``True`` for padded positions.
            * ``train_dataset[-1]`` --
              integer-encoded activity suffix labels,
              shape ``(N, W)``, right-padded with zeros after END token.

        num_categoricals_pref : int
            Number of categorical features in each prefix event token,
            including the activity label.  The activity label tensor is
            always stored at index ``num_categoricals_pref - 1``.
        """
        pref_act_tensor = train_dataset[num_categoricals_pref - 1]  # (N, W)
        padding_mask    = train_dataset[num_categoricals_pref + 1]  # (N, W) bool
        suf_act_tensor  = train_dataset[-1]                          # (N, W)

        N = pref_act_tensor.shape[0]

        # Number of non-padded events per instance  (padding_mask: True = pad)
        pref_lengths = (~padding_mask).sum(dim=1).tolist()  # list[int], length N

        # 0-based index of the END token in the suffix label tensor
        end_token_mask = (suf_act_tensor == self.end_token)          # (N, W)
        end_positions  = torch.argmax(
            end_token_mask.to(torch.int64), dim=1
        ).tolist()                                                    # list[int]

        for i in tqdm(range(N), desc="Building BEST pattern tree"):
            pref_len = pref_lengths[i]
            end_pos  = end_positions[i]

            # Prefix activities: actual events only (strip right-padding)
            prefix = pref_act_tensor[i, :pref_len].tolist()

            # Suffix activities: from position 0 up to and including END
            suffix = suf_act_tensor[i, :end_pos + 1].tolist()

            # Full trace for this prefix-suffix pair
            trace = prefix + suffix

            self._add_trace(trace)

        self.fitted = True

        # Cache most-common activity for O(1) unigram fallback
        if self.unigram_counts:
            self._unigram_most_common = max(
                self.unigram_counts, key=self.unigram_counts.get
            )

    def _add_trace(self, trace):
        """Register all n-gram patterns from a single complete trace.

        Parameters
        ----------
        trace : list of int
            Complete sequence of integer-encoded activity labels
            (no padding zeros), ending with the END token.
        """
        trace_len = len(trace)
        for pos in range(trace_len):
            next_act = trace[pos]
            # Unigram count (0-order context)
            self.unigram_counts[next_act] += 1
            # Higher-order context counts
            max_k = min(pos, self.max_context_length)
            for k in range(1, max_k + 1):
                context = tuple(trace[pos - k:pos])
                self.ngram_counts[context][next_act] += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next(self, current_sequence):
        """Predict the single next activity given the current sequence.

        Uses the longest matching context found in the n-gram table,
        backing off to shorter contexts and finally to the unigram
        distribution if no match is found.

        Parameters
        ----------
        current_sequence : list of int
            The current sequence of activity labels observed so far
            (may contain leading padding zeros, which are stripped).

        Returns
        -------
        int
            Integer index of the predicted next activity.
        """
        # Strip padding tokens (index 0) from the context
        seq = [a for a in current_sequence if a != 0]
        max_k = min(len(seq), self.max_context_length)

        # Longest-context-first back-off search
        for k in range(max_k, 0, -1):
            context = tuple(seq[-k:])
            if context in self.ngram_counts:
                counts = self.ngram_counts[context]
                return max(counts, key=counts.get)

        # Unigram fallback
        return self._unigram_most_common

    def predict_suffix(self, prefix_activities, window_size):
        """Generate the complete activity suffix for a given prefix.

        Activities are predicted autoregressively (one step at a time)
        using greedy argmax decoding.  Generation stops as soon as the
        END token is predicted or ``window_size`` steps have been
        completed.  If the END token is not predicted within
        ``window_size`` steps it is appended at the last position, and
        the output is right-padded with zeros to length ``window_size``.

        Parameters
        ----------
        prefix_activities : list of int
            Integer-encoded activity labels of the observed prefix events
            (no padding zeros).
        window_size : int
            Maximum (and target) output length ``W`` used throughout the
            dataset.

        Returns
        -------
        list of int
            Predicted activity suffix of exactly length ``window_size``,
            right-padded with zeros after the END token.
        """
        current = list(prefix_activities)
        predicted_suffix = []

        for _ in range(window_size):
            next_act = self.predict_next(current)
            predicted_suffix.append(next_act)
            current.append(next_act)
            if next_act == self.end_token:
                break

        # Enforce END token at the last position if not yet predicted
        if predicted_suffix and predicted_suffix[-1] != self.end_token:
            if len(predicted_suffix) < window_size:
                predicted_suffix.append(self.end_token)
            else:
                # window_size already reached without END: overwrite last
                predicted_suffix[-1] = self.end_token

        # Right-pad with zeros to exactly window_size
        while len(predicted_suffix) < window_size:
            predicted_suffix.append(0)

        return predicted_suffix[:window_size]
