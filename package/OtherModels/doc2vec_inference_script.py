import gensim
from helpers.utilities import calculate_cosine_similarity


def infer_doc2vec(text_1, text_2):

    # Loading the model trained in the above script
    model = gensim.models.Word2Vec.load(
        "./package/OtherModels/Models/PennDoc2Vec_model.bin")

    # Obtaining the sentence vectors for each sentence. Each sentence is first preprocessed to get the
    # list of tokens and then sent to the infer_vector method to get the sentence vectors.
    sentence_vector_1 = model.infer_vector(
        gensim.utils.simple_preprocess(text_1))
    sentence_vector_2 = model.infer_vector(
        gensim.utils.simple_preprocess(text_2))

    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = calculate_cosine_similarity(sentence_vector_1,
                                                   sentence_vector_2)[0][0]

    return('Cosine Similarity between _{0}_ and _{1}_ is **{2}**'.format(text_1,
                                                                         text_2,
                                                                         similarity_score))
