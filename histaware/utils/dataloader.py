import re
import pandas as pd
import torch
import torchtext
import glob
import numpy as np

from utils.tokenize_utils import get_nl_tokenizer, build_vocab
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)


from sklearn.model_selection import train_test_split

#MIN_WORD_FREQUENCY = 50

def cleanup_text(texts):
    cleaned_text = []
    for text in texts:
        # remove punctuation
        text = re.sub('[!#?,.:";]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)
        cleaned_text.append(text)
    return cleaned_text


def load_data(data_dir):
    data_files = glob.glob(data_dir)
    csv_data = []

    for f in data_files:
        d = pd.read_csv(f)
        csv_data.append(d)

    df = pd.concat(csv_data).reset_index()

    return df[['text','labels']]
#

# def get_nl_tokenizer():
#     """
#     Documentation:
#     https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
#     """
#     #spacy.load('nl_core_news_sm')
#     tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')
#     #tokenizer = get_tokenizer("basic_english", language="en")
#     return tokenizer
#
# def build_vocab(data_iter, tokenizer):
#     """Builds vocabulary from iterator"""
#
#     vocab = build_vocab_from_iterator(
#         map(tokenizer, data_iter),
#         specials=["<unk>","<pad>"],
#         min_freq=MIN_WORD_FREQUENCY,
#     )
#     vocab.set_default_index(vocab["<unk>"])
#     return vocab


def tokenize_text(texts):
    tokenizer = get_nl_tokenizer()
    vocab = build_vocab(texts, tokenizer)

    #text_pipeline = lambda x: vocab(tokenizer(x))
    text_pipeline = lambda x: tokenizer(x)
    return text_pipeline, vocab


def tokenize(texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.

        Args:
            texts (List[str]): List of text data

        Returns:
            tokenized_texts (List[List[str]]): List of list of tokens
            word2idx (Dict): Vocabulary built from the corpus
            max_len (int): Maximum sentence length
        """

    text_tokenizer, vocab = tokenize_text(texts)
    tokenized_texts = [text_tokenizer(sent) for sent in texts]
    padded_text = pad_text(tokenized_texts)
    encoded_texts = [vocab(sent) for sent in padded_text]
    print(encoded_texts[0])
    return encoded_texts, vocab


def pad_text(texts):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """
    max_len = len(max(texts, key=len))
    padded = [sent+ ['<pad>'] * (max_len - len(sent)) for sent in texts]

    return padded


def load_pretrained_vectors(fname, vocab):
    """Load pretrained vectors and create embedding layers.

    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

#####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pretrained = torch.load(f"{fname}/model.pt", map_location=device)
    vocab_pretrained = torch.load(f"{fname}/vocab.pt")

    # embedding from first model layer
    embeddings_pretrained = list(model_pretrained.parameters())[0]
    embeddings_pretrained = embeddings_pretrained.cpu().detach().numpy()

    # normalization
    norms = (embeddings_pretrained ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    embeddings_pretrained_norm = embeddings_pretrained / norms
    print(embeddings_pretrained_norm.shape[1])


    # Initilize random embeddings
    d = embeddings_pretrained_norm.shape[1]
    embeddings = np.random.uniform(-0.25, 0.25, (vocab.__len__(), d))
    embeddings[vocab['<pad>']] = np.zeros((d,))
    embeddings[vocab['<unk>']] = np.zeros((d,))

    vocab_dict = vocab.get_stoi()
    count =0
    for word in vocab_dict:
        word_id = vocab_pretrained[word]
        if word_id == 0:
            print("Out of vocabulary word: ", word)
            continue
        count += 1
        embeddings[vocab[word]] = embeddings_pretrained_norm[word_id]

    print(embeddings.shape)
    embeddings = torch.tensor(embeddings)
    return embeddings


def data_loader(train_inputs, val_inputs, train_labels, val_labels,
                batch_size=50):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """


    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels =\
    tuple(torch.tensor(data) for data in
          [train_inputs, val_inputs, train_labels.tolist(), val_labels.tolist()])

    # Specify batch_size
    batch_size = 50

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


