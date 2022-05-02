from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import math

def tokenize_function(examples,tokenizer):
    return tokenizer(examples["text"])

def group_texts(examples,block_size):

    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train():
    
     # Path to the training corpus. We have used PennTreeBank corpus as it is small in size.
    train_path = './package/TrainingModels/ptbdataset/ptb.train.txt'
    test_path = './package/TrainingModels/ptbdataset/ptb.valid.txt'
    
    # Loading the dataset
    datasets = load_dataset('text', data_files={'train': train_path, 'validation': test_path})

    # Initializing the model checkpoint 
    model_checkpoint = "bert-base-uncased"

    # Loading the pretrained Tokenizer and transforming the dataset with special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                         num_proc=4, remove_columns=["text"],
                                         fn_kwargs={'tokenizer':tokenizer})


    block_size = tokenizer.model_max_length

    lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
    fn_kwargs = {'block_size':block_size}
    )

    # Loading the model for Causal Language Modelling
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,hidden_size=768)

    # Path to save the training checkpoints and metadata
    trainer_path = './package/TrainingModels/Models/bert-base-uncased-finetuned-clm_PennTree'
    training_args = TrainingArguments(
        trainer_path,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    #     push_to_hub=True,
    )

    # Instantiating the trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    # Training/Fine tuning the bert model for CLM
    trainer.train()

    # Once the training is completed, we can evaluate our model and get its perplexity on 
    # the validation set like this:
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Saving the model for later use.
    model_path = './package/TrainingModels/Models/PennBert_model'
    trainer.save_model(model_path)

    print("Model Trained and saved at location: {0}".format(model_path))


if __name__ == "__main__":
    train()
