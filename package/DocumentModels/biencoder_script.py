from sentence_transformers import SentenceTransformer
from helpers.utilities import calculate_cosine_similarity
 
def infer_biencoder(text_1,text_2):   

    # Load a pretrained sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Encoding the sentences to their sentence vectors. BERT (and other transformer networks) 
    # output for each token in our input text an embedding. In order to create a fixed-sized 
    # sentence embedding out of this, the model applies mean pooling, i.e., the output embeddings 
    # for all tokens are averaged to yield a fixed-sized vector.
    sentence_vector_1 = model.encode(text_1).reshape(1,-1)
    sentence_vector_2 = model.encode(text_2).reshape(1,-1)
    
    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = calculate_cosine_similarity(sentence_vector_1,
                                                   sentence_vector_2)[0][0]

    return('Cosine Similarity between _{0}_ and _{1}_ is **{2}**'.format(text_1,
                                                                         text_2,
                                                                         similarity_score))
