The semantic similarity between two sentences using Word Embedding Models can be computed by following the below mentioned steps:

1. Loading a Pretrained Word Embedding model. Generally, this is a word embedding matrix with n dimensional vector for each word. The n can be 50,100,300 or more. 
1. Obtaining the word vectors for the sentences using the model of first step.
1. Computing the sentence vector using the word vectors either by averaging or SIF.
1. Compute the cosine similarity between the two sentence vectors.

In this section, we will see **Word2Vec, Glove, FastText, Spacy, Bert** models in practice.