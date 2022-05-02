import fasttext
import os


def main():

    # Path to the training corpus. We have used PennTreeBank corpus as it is small in size.
    data_folder ='./package/TrainingModels/ptbdataset'
    data_file = 'ptb.train.txt'
    corpus_file = os.path.join(data_folder, data_file)
  
    # Training the FastText model. It might take few minutes to train
    embedding_matrix = fasttext.train_unsupervised(corpus_file)

    # save the model for later use. You can load it later using fasttext.load_model()
    embedding_matrix_location = "./package/TrainingModels/Models/PennFastText_model.bin"
    embedding_matrix.save_model(embedding_matrix_location)

    print("Model Trained and saved at location: {0}".format(embedding_matrix_location))

if __name__ == "__main__":
    main()