import gensim
import numpy as np
from sklearn.manifold import TSNE  # final reduction


def get_act2vec_distance_matrix_our(log, alphabet, window_size):
    model = gensim.models.Word2Vec(sentences=log, vector_size=window_size, window=3, min_count=0, epochs=50, sg=1, negative = 0, hs = 1)

    #act2vev paper
    #model = gensim.models.Word2Vec(sentences=log, vector_size=vectorsize, window=3, min_count=0, alpha=0.025, min_alpha=0.005, epochs=10, sg=sg)

    '''  original code gensim version prob https://github.com/piskvorky/gensim/blob/3.4.0/gensim/models/word2vec.py 
    model = gensim.models.Word2Vec(sentences=log, size= vectorsize, window=3,  min_count=0)
    nrEpochs= 10
    for epoch in range(nrEpochs):
        if epoch % 2 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(sentences,len(sentences),start_alpha=0.025, epochs=nrEpochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    '''

    ''' 
    if random.randint(1,100) == 1:
        x_vals, y_vals, labels = reduce_dimensions(model)
        plot_function = plot_with_matplotlib

        plot_function(x_vals, y_vals, labels, len(alphabet))
    '''

    # Compute distances between all pairs of activities
    distances = {}
    for activity1 in alphabet:
        for activity2 in alphabet:
            distance = model.wv.distance(activity1, activity2)
            distances[(activity1, activity2)] = distance

    return distances
