from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd
import sys

from sklearn.manifold import TSNE
import plotly.graph_objects as go

sys.path.append("../")

def call_data():
    hist_dataset = HistawareDataset(csv_file='../data/1960/tm_63/*.csv')
    dataloader = DataLoader(hist_dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, len(sample_batched['date']), sample_batched['date'][0])


def get_embedding():
    folder = "weights/skipgram_WikiText2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(f"{folder}/model.pt", map_location=device)
    vocab = torch.load(f"{folder}/vocab.pt")

    # embedding from first model layer
    embeddings = list(model.parameters())[0]
    embeddings = embeddings.cpu().detach().numpy()

    # normalization
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    embeddings_norm = embeddings / norms
    print(embeddings_norm.shape)
    print(embeddings_norm[0])
    print(vocab)

def get_top_similar(word: str, topN: int = 10):
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings_norm[word_id]
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    topN_ids = np.argsort(-dists)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        topN_dict[sim_word] = dists[sim_word_id]
    return topN_dict

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #call_data()
    #get_embedding()
    folder = "../weights/cbow_histaware_63"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(f"{folder}/model.pt", map_location=device)
    vocab = torch.load(f"{folder}/vocab.pt")

    # embedding from first model layer
    embeddings = list(model.parameters())[0]
    embeddings = embeddings.cpu().detach().numpy()

    # normalization
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    embeddings_norm = embeddings / norms
    print(embeddings_norm.shape)
    print(vocab(["Nederlandse"]))
    print(type(vocab))
    #
    for word, sim in get_top_similar("kampioen").items():
        print("{}: {:.3f}".format(word, sim))