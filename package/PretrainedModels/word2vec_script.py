import numpy as np
import gensim.models.keyedvectors as word2vec

from helpers.utilities import calculate_cosine_similarity, convert_to_sentence_vector


def infer_word2vec(text_1, text_2):

    ######################################################################################################
    # Path to the downloaded Pretrained Word2Vector binary file. The GoogleNews-vectors-negative300.bin is
    # trained on Google New corpus of size 100B, having 300M vocabulary. Each word is a 300 dimensional
    # vector.
    ######################################################################################################

    vocab_vector_file = './package/PretrainedModels/models/GoogleNews-vectors-negative300.bin'

    # Loading the word embedding matrix from the binary Word2Vec file
    embeddings_matrix = word2vec.KeyedVectors.load_word2vec_format(vocab_vector_file,
                                                                   binary=True)

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
