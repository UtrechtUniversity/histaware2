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
        '--out_dir',
        type=str,
        nargs="+",
        help='list of sub directories for output'
    )

    parser.add_argument(
        '--data_split',
        type=str,
        nargs="+",
        help='list of split dates'
    )
    return parser

if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    input_dir = args.input_dir
    out_dirs = args.out_dir
    data_splits = args.data_split
    # n_out_dirs = len(out_dirs)
    # n_data_splits = len(data_splits)

    # fps = '../data/1960_fuel/*.csv'
    # dir_out1 = '../data/1960_fuel/tm_63/'
    # dir_out2 = '../data/1960_fuel/tm_67/'
    # dir_out3 = '../data/1960_fuel/tm_69/'
    # date_split1 = '1964-01-01'
    # date_split2 = '1968-01-01'

    csvs = glob.glob(input_dir)

    for c in csvs:
        data = pd.read_csv(c)

        filename = os.path.basename(c)


        fp1 = os.path.join(out_dirs[0], filename)
        fp2 = os.path.join(out_dirs[1], filename)
        fp3 = os.path.join(out_dirs[2], filename)

        data['date_'] = pd.to_datetime(data['date'], infer_datetime_format=True)
        p1 = data.loc[data['date_'] < data_splits[0]]
        if p1.shape[0]>0:
            p1.to_csv(fp1, index=False)
        p2 = data[(data['date_'] >= data_splits[0])&(data['date_'] < data_splits[1])]
        if p2.shape[0] > 0:
            p2.to_csv(fp2, index=False)
        p3 = data[data['date_'] >= data_splits[1]]
        if p3.shape[0] > 0:
            p3.to_csv(fp3, index=False)