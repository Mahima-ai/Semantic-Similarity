import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F

def calculate_cosine_similarity(input_1, input_2):
    """  
    Calculates the Cosine similarity using sklearn package.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

    Args:
        input_1 (_type_): Vector for the first text
        input_2 (_type_): Vector for the second text

    Returns:
        kernel matrix: ndarray of shape (n_samples_X, n_samples_Y)
    """
  
    return cosine_similarity(input_1.reshape(1,-1), input_2.reshape(1,-1), dense_output=True)
   

def convert_to_sentence_vector(sentence,embedding_matrix):
    """_summary_

    Args:
        sentence (_type_): _description_
        embedding_matrix (_type_): _description_
    
    Returns:
        _type_: Sentence Embedding
    """
    M = []
    for w in sentence.split():
        try:
            M.append(embedding_matrix[w])
        except:
            continue
    M = np.array(M)
    v = M.mean(axis=0)
    return v

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def bert_convert_to_sentence_vector(model_output,tokenizer):
    '''Generate Vectora for sentences.'''   
    
    sentence_embeddings = mean_pooling(model_output, tokenizer['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.detach().numpy()


def try_now(algorithm):
    st.subheader('Try {0} Now'.format(algorithm))
   
    text1 = st.text_input(label='Text Input 1',
                        value="Enter the first sentence")
    text2 = st.text_input(label='Text Input 2',
                        value="Enter the second sentence")
    button_clicked = st.button(label='Compute Similarity')
    return(text1,text2, button_clicked)

       