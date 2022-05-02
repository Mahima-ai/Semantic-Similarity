import gensim

from helpers.utilities import calculate_cosine_similarity, convert_to_sentence_vector


def infer_word2vec(text_1, text_2):

    # Loading the model trained in the above script
    model = gensim.models.Word2Vec.load(
        "./package/TrainingModels/Models/PennWord2Vec_model.bin")

    # Obtaining the Keyed Vector from the loaded model
    embeddings_matrix = model.wv

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
