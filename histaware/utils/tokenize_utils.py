from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from utils.constants import MIN_WORD_FREQUENCY

def get_nl_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    #spacy.load('nl_core_news_sm')
    tokenizer = get_tokenizer('spacy', language='nl_core_news_sm')
    #tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer


def build_vocab(data_iter, tokenizer):
    """Builds vocabulary from iterator"""

    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>","<pad>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab
