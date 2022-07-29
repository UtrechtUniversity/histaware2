from torch.utils.data import Dataset
import pandas as pd
import glob
import torch


class HistawareDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        data_files = glob.glob(data_dir)
        csv_data =[]

        for f in data_files:
            print(f)
            d = pd.read_csv(f)
            csv_data.append(d)

        df = pd.concat(csv_data).reset_index()
        # df = pd.read_csv(csv_file)

        self.articles = df['p']
        self.transform = transform

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if self.transform:
        #     sample = self.transform(sample)

        #'date','index_article',
        #return self.articles.loc[idx,['date','index_article','p']].to_dict()

        return self.articles[idx]