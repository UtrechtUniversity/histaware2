import pandas as pd
import glob
import os
import argparse

def parse_arguments():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        description="Histaware"
    )

    # File path to the data.
    parser.add_argument(
        "--input_dir",
        type=str,
        help="File path to the dataset of .csv files"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='list of sub directories for output'
    )

    return parser


def read_data(data_dir):
        df = pd.concat(map(pd.read_csv, glob.glob(data_dir)))
        return df[['text_split', 'labels']]

if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    input_dir = args.input_dir
    out_dirs = args.output_dir

    df = read_data(input_dir)
    df.loc[df['labels'] == 2, 'labels'] = 1
    df.columns = ['text','labels']

    df.to_csv(out_dirs)







