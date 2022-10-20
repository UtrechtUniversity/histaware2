from utils.preprocess import load_data, split_train_val_test, to_torch, clean_light

import numpy as np
import torch

import random
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

RANDOM_SEED =42

def tokenize_function(row):
    return tokenizer(
        row["text"],
        truncation=True,
)


def tokenize_data(raw_datasets):
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    return tokenized_datasets


def set_seed(seed_value=RANDOM_SEED):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, data_collator, tokenizer, training_dataset, validation_dataset, epochs=10, output_dir=""):
    """Train the CNN model."""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        logging_dir=output_dir +"/logging",
        load_best_model_at_end=True,
        seed=2020,
        # label_names=["label"], # check this
        disable_tqdm=False
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


# def save_model(model, model_dir):
#         """Save final model to `self.model_dir` directory"""
#         model_path = os.path.join(model_dir, "model.pt")
#         torch.save(model, model_path)
#

if __name__ == '__main__':
    set_seed()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to csv files')
    parser.add_argument('--model_name', type=str, required=True, help='model name in huggingface')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output')

    args = parser.parse_args()


    data_dir = args.data_dir
    model_name = args.model_name
    output_dir = args.output_dir

    df = load_data(data_dir)

    # Train Test Split
    train_dataset_raw, validation_dataset_raw, test_dataset_raw = split_train_val_test(df, random_state=RANDOM_SEED)

    # To torch dataset
    train_dataset = to_torch(train_dataset_raw)
    validation_dataset = to_torch(validation_dataset_raw)
    #test_dataset = to_torch(test_dataset_raw)

    # Clean text
    train_dataset_clean =clean_light(train_dataset)
    validation_dataset_clean =clean_light(validation_dataset)
    #test_dataset_clean =clean_light(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    print("Tokenize the datasets")
    # Tokenize the datasets
    training_dataset_tokenized = tokenize_data(train_dataset_clean)
    validation_dataset_tokenized = tokenize_data(validation_dataset_clean)
    #test_dataset_tokenized = tokenize_data(test_dataset)


    # Create data collator
    MAX_LENGHT=512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LENGHT)

    train(model, data_collator, tokenizer, training_dataset_tokenized, validation_dataset_tokenized, epochs=10,
          output_dir=output_dir)
    model.save_pretrained(output_dir)




