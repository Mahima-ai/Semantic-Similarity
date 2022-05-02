from transformers import BertModel, BertTokenizer
from helpers.utilities import bert_convert_to_sentence_vector, calculate_cosine_similarity, convert_to_sentence_vector


def infer_bert(text_1,text_2):

    # Path to the metadata and model trained in the above script
    model_path = './package/TrainingModels/Models/PennBert_model'

    # Loading the tokenizer trained in above script
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Loading weights of the model trained in the above script
    model = BertModel.from_pretrained(model_path)

    # Tokenizing the first sentence with the correct model-specific separators token type ids 
    # and attention mask that can be accepted by the model
    encoded_input_1 = tokenizer(text_1, return_tensors='pt')

    # Obtaining the output word vectors of the Bert model for first sentence 
    bert_output_1 = model(**encoded_input_1)

    # Tokenizing the second sentence with the correct model-specific separators token type ids 
    # and attention mask that can be accepted by the model
    encoded_input_2 = tokenizer(text_2, return_tensors='pt')

    # Obtaining the output word vectors of the Bert model for second sentence
    bert_output_2 = model(**encoded_input_2)

    # Converting word vectors to sentence vectors
    sentence_vector_1 = bert_convert_to_sentence_vector(bert_output_1,encoded_input_1)
    sentence_vector_2 = bert_convert_to_sentence_vector(bert_output_2,encoded_input_2)

    # Calculating the cosine similarity between the two sentence vectors
    similarity_score = calculate_cosine_similarity(sentence_vector_1,
                                                   sentence_vector_2)[0][0]

    return('Cosine Similarity between "{0}" and "{1}" is **{2}**'.format(text_1,
                                                                     text_2,
                                                                     similarity_score))
