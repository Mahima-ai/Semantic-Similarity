import os
import gensim
import numpy as np


def main():

    # Path to the training corpus. We have used PennTreeBank corpus as it is small in size.
    data_folder = './package/TrainingModels/ptbdataset'
    data_file = 'ptb.train.txt'

    # Training a Word2Vec model and setting values for various parameters.
    # The training might take few minutes.

    # size = 300             # Word vector dimensionality
    # min_count = 10         # Minimum word count
    # workers = 2            # Number of threads to run in parallel
    # window = 15            # Context window size
    # sample = 1e-3          # Downsample setting for frequent words

    model = gensim.models.Word2Vec(corpus_file=os.path.join(data_folder, data_file),
                                   workers=2,
                                   vector_size=300,
                                   min_count=1,
                                   window=15,
                                   sample=1e-3)

    # save the model for later use. You can load it later using Word2Vec.load()
    embedding_matrix_location = "./package/TrainingModels/Models/PennWord2Vec_model.bin"
    model.save(embedding_matrix_location)

    print("Model Trained and saved at location: {0}".format(embedding_matrix_location))


if __name__ == "__main__":
    main()
