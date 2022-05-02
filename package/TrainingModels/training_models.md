In the previous section we saw how we can leverage the pretrained models to get the word embeddings and later use those embeddings to obtain sentence embeddings and compute the cosine similarity between the sentences.
These models work well for the words in their vocabulary but for the out of vocabulary words (domain specific) they fail.

So in this section, we will learn to train the models on the domain of our choice so that we are able to obtain embeddings for the words of our domain. For study purpose and to minimize the training time, we will use the smallest dataset i.e. **PennTree Bank** dataset. The dataset is located under the package/TrainingModels/ptbdataset. You can use the dataset of your choice. Also, the inference script for the trained model is provided.

Training steps for the following models is provided:

1. Word2Vec 
1. Glove
1. FastText
1. Bert
