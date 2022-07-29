import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
#from torchtext.data import to_map_style_dataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils.word2vec.histaware_dataset import HistawareDataset
#from torchtext.datasets import WikiText2
#import spacy
#from utils.tokenize_utils import get_nl_tokenizer, build_vocab
import numpy as np
from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

#spacy.load('nl_core_news_sm')
def get_nl_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    #spacy.load('nl_core_news_sm')
    tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')
    #tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

#
def get_data_iterator(data_dir, ds_type, shuffle, validation_split, random_seed):

    dataset = HistawareDataset(data_dir=data_dir)

    print('type(dataset)',type(dataset))
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    print('dataset_size',dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    sampler = SubsetRandomSampler(train_indices) if (ds_type == 'train') else SubsetRandomSampler(val_indices)
    print('sampler',sampler)
    # data_iter = to_map_style_dataset(dataset)
    # return data_iter, sampler

    return dataset, sampler

# def get_data_iterator(data_dir, ds_type, shuffle, validation_split, random_seed):
#
#
#     data_iter = WikiText2(root=data_dir, split=(ds_type))
#     data_iter = to_map_style_dataset(data_iter)
#     return data_iter



def build_vocab(data_iter, tokenizer):
    """Builds vocabulary from iterator"""

    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter.articles),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(
    data_dir, model_name, ds_type, batch_size, shuffle, validation_split, random_seed, vocab=None
):

    data_iter, sampler = get_data_iterator(data_dir, ds_type, shuffle, validation_split, random_seed)
    tokenizer = get_nl_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)

    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
        sampler=sampler
    )
    return dataloader, vocab
