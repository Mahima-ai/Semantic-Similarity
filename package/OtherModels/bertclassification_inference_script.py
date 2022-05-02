from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np

def inference_bert(text_1,text_2):

    # Path to the model trained in the above script
    pt_save_directory = './package/OtherModels/Models/trained_bert_classification_model'
    
    # Loading the tokenizer trained in the above script
    tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
    
    # Loading the model trained in the above script
    pt_model = AutoModelForSequenceClassification.from_pretrained(pt_save_directory)
    
    # Preprocessing and tokenizing the sentences
    tokens = tokenizer(text_1,text_2,return_tensors="pt")
    
    # Generating predictions for the two sentences. Here, the prediction is a similarity score.
    predictions = pt_model(**tokens)

    # Obtaining the similarity classification from the predictions
    pred = torch.argmax(predictions.logits, axis=1).item()
    
    return('Cosine Similarity between _{0}_ and _{1}_ is classified in class  **{2}**'.format(text_1,
                                                                         text_2,
                                                                         pred))

