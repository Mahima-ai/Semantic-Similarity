from transformers import BertTokenizer, BertModel

from helpers.utilities import bert_convert_to_sentence_vector, calculate_cosine_similarity


def infer_bert(text_1, text_2):

    ###################################################################################################
    # Loading the tokenizer and bert-base-uncased pretrained model from the HuggingFace library. The 
    # complete list of pretrained models is available at https://huggingface.co/models
    ###################################################################################################

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    # Obtaining the output word vectors of the Bert model for first sentence
    encoded_input_1 = tokenizer(text_1, return_tensors='pt')
    bert_output_1 = model(**encoded_input_1)

    # Obtaining the output word vectors of the Bert model for second sentence
    encoded_input_2 = tokenizer(text_2, return_tensors='pt')
    bert_output_2 = model(**encoded_input_2)

    # Converting word vectors to sentence vectors
    sentence_vector_1 = bert_convert_to_sentence_vector(bert_output_1,encoded_input_1)
    sentence_vector_2 = bert_convert_to_sentence_vector(bert_output_2,encoded_input_2)

    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = calculate_cosine_similarity(sentence_vector_1,
                                                   sentence_vector_2)[0][0]

    return('Cosine Similarity between "{0}" and "{1}" is {2}'.format(text_1,
                                                                     text_2,
                                                                     similarity_score))
