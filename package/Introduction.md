Semantic similarity is one of the major NLP advances that finds its use in Information Retreival System. Semantic similarity helps to identify whether the given sentences are similar or dissimilar to each other. To find the degree of similarity, the cosine-similarity measure is widely used. 

As we all know, the machine learning algorithms process numbers and not text. So to work with text, we need to convert text to their numeric representations. These representations are usually vectors of real numbers, whereby the vectors can be either sparse or dense. In order to compute the semantic similarity between two sentences, the cosine similarity of these vector representations is calculated. The resulting value lies between 0 and 1. The higher the value, the more similar are the documents.  

The commonly used traditional algorithms for text-based features are **Bag of Words**, **tf-idf**. These models are based on the frequency of words in the document and neither contain information about the order of words nor their meaning. The word embedding models like **Word2Vec**, **Glove** and **FastText** provide vectors for each word of the sentence. To compute the similarity between sentences we have to convert these word embeddings to sentence embedding to get the vector representation of the sentence. The simplest method is to take out the average of all the word vectors. The same has been implemented in this project as under:

```
def convert_to_sentence_vector(sentence,embedding_matrix):
    
    M = []
    for w in sentence.split():
        try:
            M.append(embedding_matrix[w])
        except:
            continue
    M = np.array(M)
    v = M.mean(axis=0)
    return v
```

To convert the **Bert** word vectors into sentence vector we will use mean pooling as under:

```
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def bert_convert_to_sentence_vector(model_output,tokenizer):
    
    sentence_embeddings = mean_pooling(model_output, tokenizer['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.detach().numpy()
```

The next set of algorithms that directly provide sentence embeddings are the **Universal Sentence Encoder**, **Spacy**, **Biencoder** models of **sentence-transformer**. If we move one level up we have **Paragraph Vectors** also known as **Doc2vec** algorithm which provides paragraph/document level embeddings. 

