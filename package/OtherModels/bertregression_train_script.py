from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding# Importing the dataset¶
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def tokenize_fun(row,tokenizer):
    return tokenizer(row['sentence1'],row['sentence2'],truncation=False)

def train():

    # Lets start by importing the dataset from huggingface library. The dataset can be found at 
    # https://huggingface.co/datasets/stsb_multi_mt. 
    # We will also rename the similarity_score column to labels as the Trainer API understands only 
    # the labels column.
    dataset = load_dataset("stsb_multi_mt", name="en")
    dataset = dataset.rename_column('similarity_score','labels')

    #########################
    # Tokenize the sentences
    #########################

    # First we load a pretrained Bert tokenizer and preprocess the raw sentences. 
    # The tokenizer takes as input the raw sentences and outputs the input_id, token_type_ids and 
    # attention_mask (the numeric form) of these sentences. Once, we have input id, token_type_ids and 
    # attention_mask, we can easily train the bert model. Here, we also use a collate function. This 
    # function applies the correct amount of padding to the samples of the dataset we want to batch together. 
    # The Transformers library provides us with such a function via DataCollatorWithPadding. It takes a 
    # tokenizer when we instantiate it (to know which padding token to use, and whether the model expects 
    # padding to be on the left or on the right of the inputs) and will do everything we need.

    checkpoint = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = dataset.map(tokenize_fun, batched=True, fn_kwargs = {'tokenizer':tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ###########
    # Training
    ###########

    # We will train the BERT model using the Trainer API. The first step before we can define our Trainer is to 
    # define a TrainingArguments class that will contain all the hyperparameters the Trainer will use for 
    # training and evaluation. The only argument we have to provide is a directory where the trained model will 
    # be saved, as well as the checkpoints along the way. The second step is to define our model. We will use 
    # the AutoModelForSequenceClassification class, with one label as this is a regression problem. Once we have
    # our model, we can define a Trainer by passing it all the objects constructed up to now — the model, the 
    # training_args, the training and validation datasets, our data_collator, and our tokenizer.

    training_args = TrainingArguments("sts-bert-trainer")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    ###################
    # Saving the model
    ###################

    # We will save the model so that we can use it later for inference.

    trainer.save_model('./package/OtherModels/Models/trained_bert_regression_model')

    #############
    # Evaluation
    #############

    # To evaluate our model we make predictions for the test dataset and use the scikit-learn's 
    # mean_squared_error method to calculate the rmse and r2_score method to calculate the R-square.

    predictions = trainer.predict(tokenized_datasets["test"])

    rmse = mean_squared_error(tokenized_datasets["test"]['labels'],
                            predictions.predictions.flatten(),
                            squared=False)

    print('RMSE for the Bert Regression Similarity Model is: ',rmse)

    r2 = r2_score(tokenized_datasets["test"]['labels'],
                predictions.predictions.flatten())

    print('R2 for the Bert Regression Similarity Model is: ',r2)

if __name__=='__main__':
    train()