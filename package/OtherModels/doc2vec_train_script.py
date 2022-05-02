import os
import gensim


def generate_TaggedDocument(corpus):
    """ Takes sentences as input and returns the list
        of tagged sentences

    Args:
        corpus (str): sentences

    Returns:
        list: list of tagged sentences
    """
    list_of_tagged_docs = []
    with open(corpus) as fp:
        for i, line in enumerate(fp):
            tokens = gensim.utils.simple_preprocess(line)
            # For training data, add tags
            list_of_tagged_docs.append(
                gensim.models.doc2vec.TaggedDocument(tokens, [i]))
    return list_of_tagged_docs


def main():

    # Path to the training corpus. We have used PennTreeBank corpus as it is small in size.
    data_folder = './package/TrainingModels/ptbdataset'
    data_file = 'ptb.train.txt'

    # Doc2Vec works on the TaggedDocuments(List of strings) instead of sentences directly. Hence,
    # converting the sentences to TaggedDocuments.
    train_corpus = generate_TaggedDocument(
        os.path.join(data_folder, data_file))

    # Creating a doc2vec object and setting the parameters for training
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50,
                                          min_count=2,
                                          epochs=40)

    # Build the vocabulary
    model.build_vocab(train_corpus)

    # Train the model for the train_corpus
    model.train(train_corpus,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    # save the model for later use. You can load it later using Word2Vec.load()
    model_location = "./package/OtherModels/Models/PennDoc2Vec_model.bin"
    model.save(model_location)

    print("Model Trained and saved at location: {0}".format(model_location))

if __name__ == "__main__":
    main()
