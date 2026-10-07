import math
import operator
from queue import PriorityQueue

import tqdm
from model.metric import levenshtein_similarity
from math import log
import torch
import os
#from multiprocessing import Pool
import torch.multiprocessing as mp
from joblib import Parallel, delayed


class BeamSearchNode(object):
    def __init__(self, dec_state, previous_node, dec_X, log_prob, length, eot_found):
        self.dec_state = dec_state
        self.previous_node = previous_node
        self.dec_X = dec_X
        self.log_prob = log_prob
        self.length = length
        self.eot_found = eot_found

    def eval(self, alpha=0.75):
        reward = 0
        return self.log_prob
        # return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.length < other.length

    def __le__(self, other):
        return self.length < other.length


def batched_beam_decode(net, data_iter, num_steps, beam_size, eot_token,
                        device, name, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # We load the best model parameters
    net.load_state_dict(torch.load(name + '_best-model-parameters.pt'))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len) in zip(batched_enc_X, batched_enc_valid_len):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = torch.unsqueeze(enc_X[:, -1, 0], 1)
            # Starting node
            initial_node = BeamSearchNode(dec_state, None, dec_X, 0, 0, False)

            # Decode for one step using decoder
            Y, dec_state = net.decoder(dec_X, dec_state)
            # Beam search
            values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

            # Open nodes
            open_nodes = PriorityQueue()

            for j in range(beam_size):
                current_idx = idxs[0][0][j].view(1, -1)
                current_value = values[0][0][j].item()
                new_length = initial_node.length + (1 if not initial_node.eot_found and current_idx != eot_token else 0)

                node = BeamSearchNode(dec_state, initial_node, current_idx,
                                      initial_node.log_prob + log(current_value), new_length,
                                      initial_node.eot_found or current_idx == eot_token)

                score = -node.eval()
                open_nodes.put((score, node))

            for _ in range(1, num_steps):
                new_nodes = PriorityQueue()
                for j in range(beam_size):
                    _, current_node = open_nodes.get()
                    dec_X = current_node.dec_X
                    dec_state = current_node.dec_state

                    # Decode for one step using decoder
                    Y, dec_state = net.decoder(dec_X, dec_state)
                    # Beam search
                    values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                    for j in range(beam_size):
                        current_idx = idxs[0][0][j].view(1, -1)
                        current_value = values[0][0][j].item()
                        new_length = current_node.length + (
                            1 if not current_node.eot_found and current_idx != eot_token else 0)
                        node = BeamSearchNode(dec_state, current_node, current_idx,
                                              current_node.log_prob + log(current_value), new_length,
                                              current_node.eot_found or current_idx == eot_token)
                        score = -node.eval()
                        new_nodes.put((score, node))
                for j in range(beam_size):
                    open_nodes.put(new_nodes.get())
            # We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = open_nodes.get()
            for i in range(num_steps - 1, -1, -1):
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred)
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds


class BeamSearchNodeOptimized(object):
    def __init__(self, dec_state, previous_node, dec_X, log_prob, length, eot_found, max_length, enc_valid_len,
                 attention_weights):
        self.dec_state = dec_state
        self.previous_node = previous_node
        self.dec_X = dec_X
        self.log_prob = log_prob
        self.length = length
        self.eot_found = eot_found
        self.max_length = max_length
        self.enc_valid_len = enc_valid_len
        self.attention_weights = attention_weights

    def eval(self, beam_type, alpha=0.65, beta=0.65):
        reward = 0
        if beam_type == "beam":
            return self.log_prob
        elif beam_type == "beam_length_normalized":
            # Following https://arxiv.org/pdf/1609.08144.pdf is not a product but a division (log(P(Y|X))/lp(Y))
            return self.log_prob / (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha))
        elif beam_type == "beam_monteagudo":
            return self.log_prob * (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha))

        elif beam_type == "beam_length_normalized_coverage":
            # Use attention_weights to compute a coverage penalization
            coverage_penalization = [0.0 for i in range(self.enc_valid_len)]
            current_node = self
            if self.previous_node is None:
                scalar_penalization = 0.0
            else:
                while current_node.previous_node is not None:
                    attention_weights = current_node.attention_weights[0][0, 0, :].tolist()
                    coverage_penalization = [coverage_penalization[i] + attention_weights[i] for i in
                                             range(len(coverage_penalization))]
                    current_node = current_node.previous_node
                coverage_penalization = [log(min(coverage_penalization[i], 1.0) + 0.00001) for i in
                                         range(len(coverage_penalization))]
                # Coverage penalization
                scalar_penalization = beta * sum(coverage_penalization)
            # Final score
            return self.log_prob / (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha)) + scalar_penalization

        elif beam_type == "beam_length_normalized_with_penalty":
            return (self.log_prob / (math.pow(5 + self.length, alpha) / math.pow(5 + 1, alpha))) - 0.3 * (
                        self.max_length / (self.length + 1))
        else:
            raise ValueError("Unknown beam type")

    def __lt__(self, other):
        return self.length < other.length

    def __le__(self, other):
        return self.length < other.length


def setup_reproducibility():
    # Set seeds for reproducibility
    # Call this function just before predicting. Otherwise, the predictions will be different if we train
    # and test and if we only test using the saved weights
    import torch
    import random
    import numpy as np
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

def do_beam_search_1(args):
    with torch.no_grad():
        (enc_X, enc_valid_len, grnn_X, net, max_len, postprocessing_type, num_steps, eot_token, beam_size) = args

        enc_X = enc_X.unsqueeze(0)
        enc_valid_len = enc_valid_len.unsqueeze(0)
        grnn_X = grnn_X.unsqueeze(0)

        # We get the outputs of the encoder
        if hasattr(net.encoder, "rnn"):
            net.encoder.rnn.flatten_parameters()
        enc_outputs = net.encoder(enc_X, grnn_X)
        if hasattr(net.encoder, "rnn"):
            net.decoder.rnn.flatten_parameters()
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
        # We prepare the first output for the decoder
        dec_X = torch.unsqueeze(enc_X[:, -1, 0], 1)
        # Starting node
        initial_node = BeamSearchNodeOptimized(dec_state, None, dec_X, 0, 0, False, max_length=max_len,
                                               enc_valid_len=enc_valid_len, attention_weights=None)

        open_nodes = PriorityQueue()
        end_nodes = []
        number_required = 1

        open_nodes.put((-initial_node.eval(postprocessing_type), initial_node))

        for _ in range(num_steps):
            current_score, current_node = open_nodes.get()
            dec_X = current_node.dec_X
            dec_state = current_node.dec_state

            if dec_X.item() == eot_token and current_node.previous_node != None:
                end_nodes.append((current_score, current_node))
                if len(end_nodes) >= number_required:
                    break
                else:
                    continue

            # Decode for one step using decoder
            Y, dec_state = net.decoder(dec_X, dec_state)
            # Beam search
            values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

            for j in range(beam_size):
                current_idx = idxs[0][0][j].view(1, -1)
                current_value = values[0][0][j].item()
                new_length = current_node.length + (
                    1 if not current_node.eot_found and current_idx != eot_token else 0)
                node = BeamSearchNodeOptimized(dec_state, current_node, current_idx,
                                               current_node.log_prob + log(current_value), new_length,
                                               current_node.eot_found or current_idx == eot_token,
                                               max_length=max_len, enc_valid_len=enc_valid_len, attention_weights=None)
                score = -node.eval(postprocessing_type)
                open_nodes.put((score, node))

        if (len(end_nodes) == 0):
            end_nodes = [open_nodes.get() for _ in range(number_required)]
        # We prepare an aditional variable for storing the results
        current_pred = []
        _, best_node = sorted(end_nodes, key=operator.itemgetter(0))[0]
        while best_node.previous_node is not None:
            current_pred = [best_node.dec_X.item()] + current_pred
            best_node = best_node.previous_node
        return current_pred + [eot_token] * (num_steps - len(current_pred))


def parallel_beam_search_1(net, data_iter, num_steps, beam_size, eot_token,
                                  device, name, attention_enabled, postprocessing_type, max_len,
                                  save_attention_weights=False, load_net=True):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if load_net:
        if attention_enabled:
            net.load_state_dict(torch.load(os.path.join("./results/models", name + '_best-model-parameters.pt')))
        else:
            net.load_state_dict(
                torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))


    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    #net.share_memory()
    for batch in tqdm.tqdm(data_iter, total=len(data_iter)):
        batched_enc_X, batched_enc_valid_len, _, _, batched_grnn_X = [x.to(device) for x in batch]

        args = [(enc_X, enc_valid_len, grnn_X, net, max_len, postprocessing_type, num_steps, eot_token, beam_size) for (enc_X, enc_valid_len, grnn_X) in zip(batched_enc_X, batched_enc_valid_len, batched_grnn_X)]

        # This copies the model verbatim as many times as there are processes: high GPU memory usage
        results = Parallel(n_jobs=6)(delayed(do_beam_search_1)(arg) for arg in args)
        preds.append(torch.Tensor(results).to(torch.int))

    return preds


def superoptimized_batch_beam_search(net, data_iter, num_steps, beam_size, eot_token,
                                  device, name, attention_enabled, postprocessing_type, max_len,
                                  save_attention_weights=False, load_net=True):

    preds = []
    for batch in data_iter:
        batch_size = batch[0].shape[0]
        batched_enc_X, batched_enc_valid_len, _, _, batched_grnn_X, A = [x.to(device) for x in batch]
        enc_outputs = net.encoder(batched_enc_X, batched_grnn_X, A)
        dec_state = net.decoder.init_state(enc_outputs, batched_enc_valid_len)

        batch_size = dec_state[0].shape[0]

        beam_scores = torch.zeros((batch_size, beam_size)).to(device)
        beam_scores[:, 1:] = -1e9
        first_decoder_token = batched_enc_X[:, -1, 0].unsqueeze(-1).unsqueeze(-1)
        first_decoder_token = first_decoder_token.repeat(1, beam_size, 1).to(device)

        #beam_seqs = torch.full((batch_size, beam_size, 1), eot_token).to(device)
        beam_seqs = first_decoder_token

        first_state = dec_state[0].repeat(beam_size, 1, 1)
        second_state = dec_state[1].repeat(1, beam_size, 1)
        third_state = dec_state[2].repeat(beam_size)

        for i in range(max_len):

            flatten_beam_seqs = beam_seqs.view(batch_size * beam_size, -1)
            #flatten_beam_masks = beam_masks.view(batch_size * beam_size, -1)


            Y, _ = net.decoder(flatten_beam_seqs, (first_state, second_state, third_state))
            unflatten_Y = Y.view(batch_size, beam_size, Y.shape[1], Y.shape[2])

            relevant_score = unflatten_Y[:, :, i, :]
            scores = relevant_score.softmax(dim=-1)
            scores = beam_scores.unsqueeze(-1) + scores.log()

            # TODO: this length penalty is not the same as in the other case because it does not stop at EOC,
            # i.e, it computes the penalty for the remaining sequence after the first EOC (more EOCs).
            length_penalty = ((5 + (i+1)) ** 0.65) / ((5 + 1) ** 0.65)
            scores = scores / length_penalty
            scores = scores.view(batch_size, -1)

            topk_scores, topk_indices = torch.topk(scores, beam_size, dim=-1)

            # Calculate the index of the corresponding candidate sequence in the beam for each of the top k scores
            # (tell me which beam the top k scores belong to)
            beam_index = topk_indices // scores.size(-1)
            # Calculate the index of the corresponding token in the candidate sequence for each of the top k scores
            # (tell me which token in the candidate sequence the top k scores belong to)
            token_index = topk_indices % scores.size(-1)

            indexes = beam_index.unsqueeze(-1).repeat(1, 1, flatten_beam_seqs.shape[-1])
            beam_seqs = torch.gather(flatten_beam_seqs.view(batch_size, beam_size, -1), 1, indexes)
            beam_seqs = torch.cat([beam_seqs, token_index.unsqueeze(-1)], dim=-1)

        max_score_indices = beam_scores.argmax(dim=1)
        best_seqs = beam_seqs[torch.arange(batch_size), max_score_indices]
        # The first token corresponds to the first state of the encoder and it is not a prediction
        preds.append(best_seqs[:, 1:])
    return preds



def batched_beam_decode_optimized(net, data_iter, num_steps, beam_size, eot_token,
                                  device, name, attention_enabled, postprocessing_type, max_len,
                                  save_attention_weights=False, load_net=True):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if load_net:
        if attention_enabled:
            net.load_state_dict(torch.load(os.path.join("./results/models", name + '_best-model-parameters.pt')))
        else:
            net.load_state_dict(
                torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    for batch in data_iter:
        batch_pred = []
        batched_enc_X, batched_enc_valid_len, _, _, batched_grnn_X = [x.to(device) for x in batch]
        for (enc_X, enc_valid_len, grnn_X) in zip(batched_enc_X, batched_enc_valid_len, batched_grnn_X):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            grnn_X = grnn_X.unsqueeze(0)

            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, grnn_X)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = torch.unsqueeze(enc_X[:, -1, 0], 1)
            # Starting node
            initial_node = BeamSearchNodeOptimized(dec_state, None, dec_X, 0, 0, False, max_length=max_len, enc_valid_len=enc_valid_len, attention_weights=None)

            open_nodes = PriorityQueue()
            end_nodes = []
            number_required = 1

            open_nodes.put((-initial_node.eval(postprocessing_type), initial_node))

            for _ in range(num_steps):
                current_score, current_node = open_nodes.get()
                dec_X = current_node.dec_X
                dec_state = current_node.dec_state

                if dec_X.item() == eot_token and current_node.previous_node != None:
                    end_nodes.append((current_score, current_node))
                    if len(end_nodes) >= number_required:
                        break
                    else:
                        continue

                # Decode for one step using decoder
                Y, dec_state = net.decoder(dec_X, dec_state)
                # Beam search
                values, idxs = Y.softmax(dim=2).topk(beam_size, dim=2)

                for j in range(beam_size):
                    current_idx = idxs[0][0][j].view(1, -1)
                    current_value = values[0][0][j].item()
                    new_length = current_node.length + (
                        1 if not current_node.eot_found and current_idx != eot_token else 0)
                    node = BeamSearchNodeOptimized(dec_state, current_node, current_idx,
                                                   current_node.log_prob + log(current_value), new_length,
                                                   current_node.eot_found or current_idx == eot_token,
                                                   max_length=max_len, enc_valid_len=enc_valid_len, attention_weights=None)
                    score = -node.eval(postprocessing_type)
                    open_nodes.put((score, node))

            if (len(end_nodes) == 0):
                end_nodes = [open_nodes.get() for _ in range(number_required)]
            # We prepare an aditional variable for storing the results
            current_pred = []
            _, best_node = sorted(end_nodes, key=operator.itemgetter(0))[0]
            while best_node.previous_node is not None:
                current_pred = [best_node.dec_X.item()] + current_pred
                best_node = best_node.previous_node
            batch_pred.append(current_pred + [eot_token] * (num_steps - len(current_pred)))
        preds.append(torch.Tensor(batch_pred).to(torch.int))
    return preds


# More optimal than versions below
def predict_seq2seq_test(net, data_iter, num_steps,
                         device, name, attention_enabled, batch_size, postprocessing_strategy, eot_token,
                         save_attention_weights=False):
    """Predict for sequence to sequence."""
    # We load the best model parameters
    setup_reproducibility()
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    for batch in data_iter:
        enc_X, enc_valid_len, _, _ = [x.to(device) for x in batch]
        # We get the outputs of the encoder
        enc_outputs = net.encoder(enc_X, enc_valid_len)
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
        # We prepare the first output for the decoder
        dec_X = torch.unsqueeze(enc_X[:, -1, 0], 1)
        # We prepare an aditional variable for storing the results
        pred = torch.empty((enc_X.shape[0], num_steps + 1), dtype=torch.int)
        # We iterate over the steps in the decoder
        for i in range(num_steps):
            Y, dec_state = net.decoder(dec_X, dec_state)
            if postprocessing_strategy == "argmax":
                dec_X = Y.argmax(dim=2)
            elif postprocessing_strategy == "random":
                probas = Y[:, 0, :]
                dec_X = torch.multinomial(probas.softmax(dim=-1), 1)[:, 0]
                dec_X = dec_X.unsqueeze(1)
            else:
                raise ValueError("Unknown postprocessing strategy")
            pred[:, i + 1] = dec_X.squeeze()
        preds.append(pred[:, 1:])
    return preds


def predict_seq2seq(net, data_iter, num_steps,
                    device, name, attention_enabled, batch_size, postprocessing_strategy, eot_token,
                    save_attention_weights=False):
    setup_reproducibility()
    """Predict for sequence to sequence."""
    # We load the best model parameters
    if attention_enabled:
        net.load_state_dict(torch.load(os.path.join("./results_attention/models", name + '_best-model-parameters.pt')))
    else:
        net.load_state_dict(
            torch.load(os.path.join("./results_no_attention/models", name + '_best-model-parameters.pt')))
    # Set `net` to eval mode for inference
    net.eval()
    # We get predictions for each batch
    preds = []
    final_preds = []
    for batch in data_iter:
        batch_enc_X, batch_enc_valid_len, _, _ = [x.to(device) for x in batch]
        for enc_X, enc_valid_len in zip(torch.unbind(batch_enc_X), torch.unbind(batch_enc_valid_len)):
            enc_X = enc_X.unsqueeze(0)
            enc_valid_len = enc_valid_len.unsqueeze(0)
            # print("Enc X: ", enc_X.shape)
            # print("Enc valid len: ", enc_valid_len.shape)
            # We get the outputs of the encoder
            enc_outputs = net.encoder(enc_X, enc_valid_len)
            dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
            # We prepare the first output for the decoder
            dec_X = enc_X[:, -1, 0].reshape(-1, 1).expand(-1, num_steps + 1).detach().clone()
            # We iterate over the steps in the decoder
            for i in range(num_steps):
                Y, dec_state = net.decoder(dec_X[:, :num_steps], dec_state)
                if postprocessing_strategy == "argmax":
                    curr_pred = Y.argmax(dim=2)[:, i]
                    dec_X[:, i + 1] = curr_pred
                elif postprocessing_strategy == "random":
                    curr_pred = torch.multinomial(Y.softmax(dim=2)[:, i], 1)[:, 0]
                    dec_X[:, i + 1] = curr_pred
                else:
                    raise ValueError("Unknown postprocessing strategy")
                if curr_pred.item() == eot_token:
                    break

            eot_tensor = torch.tensor([eot_token] * (num_steps - (i))).unsqueeze(dim=0)
            concat_tensor = torch.cat((dec_X[:, 1:i + 1].to(torch.int).cpu(), eot_tensor), dim=-1)
            preds.append(concat_tensor)

    # We need to compact the predictions in chucks of the size of the batch in order to calculate
    # the similarity correctly.
    for n_batch, batch in enumerate(data_iter):
        my_arr = []
        for i in range(len(batch)):
            my_arr.append(preds[n_batch * batch_size + i])
        final_preds.append(torch.cat(my_arr, dim=0))

    return final_preds
