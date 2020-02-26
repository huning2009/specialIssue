# env: Python 3.7.4
# Run this script locally, not on server
# Lyndon on Feb. 25, 2020
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-fp', type=str, nargs='+', help='Parameter: the absolute path of the uncoded data')
args = parser.parse_args()


class split:
    def __init__(self, file_path):
        self.uncoded = file_path[0]

    def split_data(self):
        data = pd.read_csv(self.uncoded, header=0, sep=',')
        rowNum = data.shape[0]
        baseline = int(rowNum / 200000)
        cursor = 0
        for idx in range(0, baseline):
            if idx <= baseline - 2:
                data.iloc[cursor: cursor + 200000, :].to_csv('/'.join(self.uncoded.split('/')[:-2]) +
                                                             '/uncoded_data/' +
                                                             str(cursor) + '_' + str(cursor + 200000) + '.csv',
                                                             sep = ',',
                                                             header=True,
                                                             index=None,
                                                             encoding='utf-8')
                cursor += 200000
                print('>>>>>> Process:', cursor - 200000, 'to', cursor, '...')
            else:
                data.iloc[200000 * (baseline - 1) - rowNum:, :].to_csv('/'.join(self.uncoded.split('/')[:-2]) +
                                                                       '/uncoded_data/end.csv',
                                                                       sep=',',
                                                                       header=True,
                                                                       index=None,
                                                                       encoding='utf-8')
                print('>>>>>> Process: All finished :)')

if __name__ == '__main__':
    split = split(args.fp)
    split.split_data()
