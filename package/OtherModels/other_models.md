In this section, we will go through few more models like **Document Embedding Model**, **Cross Encoder** and variations of Bert model for semantic similarity.

1. **Doc2Vec** : A Doc2Vec is a document embedding model which provides the embedding for the complete document. Using these document vectors we can compute the cosine similarity between two documents. There are two implementations of Doc2Vec in *gensim* library.:

    1. Paragraph Vector - Distributed Memory (PV-DM)
    1. Paragraph Vector - Distributed Bag of Words (PV-DBOW)


1. **Cross Encoder** : Cross Encoder does not output a sentence embedding rather it gives a similarity score between 0 and 1 indicating how similar are the given pair of sentences. 

1. **Bert as a Classification Model** : The Bert Model is one of the SOTA models. For a semantic similarity problem, bert can be fine tuned on a classification task. Here, the model is fine tuned on a glue benchmak ***mrpc*** dataset. Once, the model is fine tuned it is able to classify the sentences as similar sentences ( label 1) or dissimilar sentences (label 0).

1.  **Bert as a Regression Model** : For a semantic similarity problem, bert can also be fine tuned on a regression task. Here, the model is fine tuned on a ***sts*** benchmark dataset. Once, the model is fine tuned it is able to predict the degree of similarity between the given pair of sentences in a range of 0 to 5. A score of 5 indicates high similarity where as 0 indicates that the two sentences are completely dissimilar.
