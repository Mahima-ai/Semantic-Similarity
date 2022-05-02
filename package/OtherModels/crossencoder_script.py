from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

def infer_crossencoder(text_1,text_2):   

    # Load a pretrained cross-encoder transformer model. The model loaded here is stsb-roberta-large 
    # of size 1.32 GB
    model = CrossEncoder('cross-encoder/stsb-roberta-large') 

    # Compute the similarity score using the predict method.
    similarity_score = model.predict([(text_1, text_2)])
   
    return('Cosine Similarity between _{0}_ and _{1}_ is **{2}**'.format(text_1,
                                                                         text_2,
                                                                         similarity_score))
