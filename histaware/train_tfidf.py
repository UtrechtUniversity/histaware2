import argparse
import yaml
import glob
import pandas as pd

from utils.tfidf.preprocess_tfidf import train_tfidf,apply_tfidf,clean_text


def load_data(data_dir):
    data_files = glob.glob(data_dir)
    csv_data = []

    for f in data_files:
        # print(f)
        d = pd.read_csv(f)
        d = clean_text(d)
        csv_data.append(d)

    df = pd.concat(csv_data).reset_index()
    return df

def train(data_dir,test_data_dir, most_commo_keywords=False):
    '''
    Calculate IDF on the whole dataset
    '''
    #os.makedirs(config["model_dir"])
    # data_dir = config["data_dir"]
    df = load_data(data_dir)
    tfidf_transformer,cv = train_tfidf(df['p'])

    docs_test = load_data(test_data_dir)
    keywords = apply_tfidf(tfidf_transformer, cv, docs_test['p'],most_commo_keywords=most_commo_keywords)
    print(keywords)
    return keywords

    # df = train_tfidf(df['p'])
    # return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to csv files')
    parser.add_argument('--test_data_dir', type=str, required=True, help='path to test csv files')
    parser.add_argument('--common_keywords', type=bool, default=False, help='path to test csv files')

    args = parser.parse_args()

    train(args.data_dir, args.test_data_dir, args.common_keywords)