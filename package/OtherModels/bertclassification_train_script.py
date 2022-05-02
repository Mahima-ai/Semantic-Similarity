from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding# Importing the dataset¶
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np

def tokenize_fun(row,tokenizer):
    return tokenizer(row['sentence1'],row['sentence2'],truncation=False)

def train():

    # Lets start by importing the dataset from huggingface library. The dataset can be found at 
    # https://huggingface.co/datasets/glue/viewer/mrpc/train. 
    dataset = load_dataset("glue", name="mrpc")

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
    # the AutoModelForSequenceClassification class, with two labels as this is a classification problem. Once we have
    # our model, we can define a Trainer by passing it all the objects constructed up to now — the model, the 
    # training_args, the training and validation datasets, our data_collator, and our tokenizer.

    training_args = TrainingArguments("mrpc-bert-trainer")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    ###################
    # Saving the model
    ###################

    # We will save the model so that we can use it later for inference.

    trainer.save_model('./package/OtherModels/Models/trained_bert_classification_model')

    #############
    # Evaluation
    #############

    # To evaluate our model we make predictions for the test dataset and use the scikit-learn's 
    # accuracy_score method to calculate the accuracy and roc_auc_score method to calculate the AUC score.

    predictions = trainer.predict(tokenized_datasets["test"])

    # Predicting the class labels from the prediction logits
    pred = np.argmax(predictions.predictions, axis=1)

    acc = accuracy_score(tokenized_datasets["test"]['label'],
                            pred,
                            squared=False)

    print('Accuracy for the Bert Classification Similarity Model is: ',acc)

    auc = roc_auc_score(tokenized_datasets["test"]['label'],
                predictions.predictions[:,1])

    print('AUC score for the Bert Classification Similarity Model is: ',auc)

if __name__=='__main__':
    train()