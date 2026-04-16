# =============================================================================
# Based on:
# De Koninck, Pieter, Seppe Vanden Broucke, and Jochen De Weerdt.
# "act2vec, trace2vec, log2vec, and model2vec: Representation Learning for
# Business Processes." Business Process Management: 16th International
# Conference, BPM 2018, Sydney, NSW, Australia, September 9–14, 2018,
# Proceedings 16. Springer International Publishing, 2018.
# https://doi.org/10.1007/978-3-319-98648-7_18
# =============================================================================

from itertools import combinations

import gensim
import numpy as np
from sklearn.manifold import TSNE  # final reduction
from numpy.linalg import norm

def get_act2vec_distance_matrix(log, alphabet, sg, window_size):

    '''  original act2vec code with gensim version ca 3.4.0 https://github.com/piskvorky/gensim/blob/3.4.0/gensim/models/word2vec.py
    model = gensim.models.Word2Vec(sentences=log, size= vectorsize, window=3,  min_count=0)
    nrEpochs= 10
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(sentences,len(sentences),start_alpha=0.025, epochs=nrEpochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    '''

    vectorsize = 16
    # Initialize model
    model = gensim.models.Word2Vec(sentences=log, vector_size=vectorsize, window=window_size, min_count=0, workers=1, sg=sg)

    nrEpochs = 10
    alpha = 0.025
    min_alpha = 0.0001  # Minimum learning rate

    # Training loop with manual learning rate adjustment
    for epoch in range(nrEpochs):
        alpha = alpha - 0.002

        model.train(log, total_examples=model.corpus_count, epochs=nrEpochs, start_alpha=0.025)

        model.alpha -= 0.002  # Set new learning rate
        model.min_alpha = model.alpha  # Prevent further decay

    # Compute distances between all pairs of activities
    distances = {}
    for activity1 in alphabet:
        for activity2 in alphabet:
            distance = model.wv.distance(activity1, activity2)
            distances[(activity1, activity2)] = distance


    embedding_dict = {word: model.wv[word] for word in model.wv.index_to_key}

    # 1. Manually compute cosine distances between every pair of words
    manual_distances = {}
    for word1 in alphabet:
        for word2 in alphabet:
            vec1 = embedding_dict[word1]
            vec2 = embedding_dict[word2]
            # Compute cosine similarity: (dot product) / (norms product)
            cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
            # Cosine distance is defined as 1 - cosine similarity
            cosine_distance = 1 - cos_sim
            manual_distances[(word1, word2)] = cosine_distance

    return distances, embedding_dict

'''
def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels, number_of_activities):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, number_of_activities)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()
'''