In the previous section, we saw how we can use **Word Embedding Models** to obtain word vectors. These word vectors are useful when we have to do comparison between words. But, to compare two sentence we either average these word vectors to obtain sentence vectors and then compute the cosine similarity between these sentence vectors or use Sentence Embedding Models.

The **Sentence Embedding models** directly provide the sentence vectors for the given sentences. These models are used to get the sentence-level embeddings. Some of the Sentence Embedding Models are:

1. Universal Sentence Encoder
1. Skip-thoughts
1. Infer-sent 
1. Biencoder Models 

An exhaustive list of Sentence Embedding Models (biencoder models) is available at https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models. These models can be put into action using the *sentence-transformer* library. The installation process is mentioned at https://www.sbert.net/docs/installation.html.