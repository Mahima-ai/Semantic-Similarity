import numpy as np
from helpers.utilities import convert_to_sentence_vector, calculate_cosine_similarity


def infer_glove(text_1, text_2):

    ######################################################################################################
    # Path to the downloaded pretrained Glove Vector file. The glove.6B.100d.txt is
    # trained on 6 billion tokens and each token is a 100 dimension vector. The file
    # is in the following format:
    #
    # for -0.14401 0.32554 0.14257 -0.099227 0.72536 0.19321 -0.24188 0.20223 -0.89599 0.15215 0.035963
    # -0.59513 -0.051635 -0.014428 0.35475 -0.31859 0.76984 -0.087369 -0.24762 0.65059 -0.15138 -0.42703
    # 0.18813 0.091562 0.15192 0.11303 -0.15222 -0.62786 -0.23923 0.096009 -0.46147 0.41526 -0.30475
    # 0.1371 0.16758 0.53301 -0.043658 0.85924 -0.41192 -0.21394 -0.51228 -0.31945 0.12662 -0.3151
    # 0.0031429 0.27129 0.17328 -1.3159 -0.42414 -0.69126 0.019017 -0.13375 -0.096057 1.7069 -0.65291
    # -2.6111 0.26518 -0.61178 2.095 0.38148 -0.55823 0.2036 -0.33704 0.37354 0.6951 -0.001637 0.81885
    # 0.51793 0.27746 -0.37177 -0.43345 -0.42732 -0.54912 -0.30715 0.18101 0.2709 -0.29266 0.30834 -1.4624
    # -0.18999 0.92277 -0.099217 -0.25165 0.49197 -1.525 0.15326 0.2827 0.12102 -0.36766 -0.61275 -0.18884
    # 0.10907 0.12315 0.090066 -0.65447 -0.17252 2.6336e-05 0.25398 1.1078 -0.073074
    ########################################################################################################

    vocab_vector_file = './package/PretrainedModels/models/glove.6B.100d.txt'

    embeddings_matrix = dict()

    # Converting the vector file into a dictionary format with tokens as key and vector as value.
    with open(vocab_vector_file, 'r') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_matrix[word] = coefs

    # Converting word vectors to sentence vectors
    sentence_vector_1 = convert_to_sentence_vector(text_1,
                                                   embeddings_matrix)
    sentence_vector_2 = convert_to_sentence_vector(text_2,
                                                   embeddings_matrix)

    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = calculate_cosine_similarity(sentence_vector_1,
                                                   sentence_vector_2)[0][0]

    return('Cosine Similarity between _{0}_ and _{1}_ is **{2}**'.format(text_1,
                                                                         text_2,
                                                                         similarity_score))
