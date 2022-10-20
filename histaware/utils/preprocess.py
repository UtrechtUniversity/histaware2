import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from datasets import Dataset
from utils.clean_text import TextCleaner


def load_data(data_dir):
    data_files = glob.glob(data_dir)
    csv_data = []

    for f in data_files:
        print(f)
        d = pd.read_csv(f)
        csv_data.append(d)

    df = pd.concat(csv_data).reset_index()

    return df[['text_split','labels']]


def split_train_val_test(df,random_state):
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text_split'], df['labels'],
                                                                        test_size=.2, random_state=random_state)
    train_dataset = pd.DataFrame()
    train_dataset["text"] = train_texts
    train_dataset["labels"] = train_labels

    # Divide val into val and test"
    test_texts, val_texts, test_labels, val_labels = train_test_split(val_texts, val_labels, test_size=.5,
                                                                      random_state=random_state)
    validation_dataset = pd.DataFrame()
    validation_dataset["text"] = val_texts
    validation_dataset["labels"] = val_labels

    test_dataset = pd.DataFrame()
    test_dataset["text"] = test_texts
    test_dataset["labels"] = test_labels

    return train_dataset, validation_dataset, test_dataset


text_cleaner = TextCleaner()


def clean_function(row):
    return text_cleaner.preprocess(row['text'])


def clean_light(dataset):
    clean_dataset = dataset.map(clean_function,batched=True)
    return clean_dataset


def to_torch(pd_dataset):
    # Transform into Dataset from hf
    torch_dataset = Dataset.from_pandas(pd_dataset)
    torch_dataset = torch_dataset.rename_column('labels', 'label')
    return torch_dataset
