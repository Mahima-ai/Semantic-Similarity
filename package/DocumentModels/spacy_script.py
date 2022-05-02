import spacy


def infer_spacy(text_1, text_2):

    ###############################################################################################
    # Spacy provides four variants of the model- sg, md,lg and trf. We have used lg variant as
    # it has a large word vector table with 500k entries. The models can be downloaded using the
    # following code:
    # !python -m spacy download en_core_web_lg
    # Once the model is downloaded you can execute the following code
    ###############################################################################################

    model = "en_core_web_lg"

    # Loading the model
    nlp = spacy.load(model)

    # Create an nlp (spacy model) object for the text
    doc1 = nlp(text_1)
    doc2 = nlp(text_2)

    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = doc1.similarity(doc2)

    return('Cosine Similarity between _{0}_ and _{1}_ is **{2}**'.format(text_1,
                                                                         text_2,
                                                                         similarity_score))
