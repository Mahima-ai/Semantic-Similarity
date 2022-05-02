import tensorflow_hub as hub
from helpers.utilities import calculate_cosine_similarity
  

def infer_use(text_1,text_2):
    # Load pretrained universal sentence encoder model from tensorflow hub
    embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  
    # Obtaining the sentence vectors
    sentence_vector_1 = embedding([text_1]).numpy()
    sentence_vector_2 = embedding([text_2]).numpy()

    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = calculate_cosine_similarity(sentence_vector_1,
                                                   sentence_vector_2)[0][0]

    return('Cosine Similarity between _{0}_ and _{1}_ is **{2}**'.format(text_1,
                                                                         text_2,
                                                                         similarity_score))
