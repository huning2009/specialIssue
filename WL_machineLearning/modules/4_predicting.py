# env: Python 3.7.4
# You can run this script in server (scribe in UCD)
# Lyndon on Feb. 25, 2020
import os
import joblib
import argparse
import pandas as pd
from sklearn import preprocessing

pd.set_option('max_rows', 500)
pd.set_option('max_columns', 500)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

parser = argparse.ArgumentParser()
parser.add_argument('-fp', type=str, nargs='+', help='First parameter: the absolute file path of the column names. '
                                                     '\nSecond parameter: the absolute file folder path of the uncoded data (without slash)')
args = parser.parse_args()


class predict:
    def __init__(self, file_path):
        self.column_names_file = file_path[0]  # file
        self.uncoded_data_folder = file_path[1]  # folder

    def main(self, uncoded_data_file):  # parameter: each file in the folder
        column_names = list()
        with open(self.column_names_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                column_names.append(line.strip())
        # print(column_names)
        df = pd.DataFrame(columns=column_names)  # an empty dataframe
        # print(df)

        print('>>>>>> Now,', uncoded_data_file)
        data = pd.read_csv(uncoded_data_file, sep=',', header=0, dtype={'pid': str})  # 1. pid str
        print('>>>>>> Original data for predicting has %s rows and %s columns' % (data.shape[0], data.shape[1]))
        print('>>>>>> Number of `pcode` is: %s' % (len(set(data.pcode))))
        data.drop('scode.1', axis=1, inplace=True)  # 2. drop scode.1
        data = pd.concat([data, pd.get_dummies(data.pcode)], axis=1, join='inner')

        for column in data.columns:
            df[column] = data[column]
            df.fillna(0, inplace=True)
        # print(df)

        df.success = df.success.replace([0, 1], ['FALSE', 'TRUE'])  # 3. filter by success and confidence
        # print(df)
        print('>>>>>> `success`', [i for i in zip(list(df.success.value_counts().index),
                                                  list(df.success.value_counts().values))])
        df = df[df['success'] != 'FALSE']
        print('>>>>>> Now data has %s rows and %s columns' % (df.shape[0], df.shape[1]))
        df = df[df['confidence'] >= .85]
        print('>>>>>> Now data has %s rows and %s columns' % (df.shape[0], df.shape[1]))

        df.tag = df.tag.apply(lambda x: str(x.split('_')[1]).strip())  # 4. tag split
        print('>>>>>> `tag`', [i for i in zip(list(df.tag.value_counts().index),
                                              list(df.tag.value_counts().values))])

        df.drop('coder_result', axis=1, inplace=True)
        # print(df)

        min_max_scaler.fit(df[['pose_Tx', 'pose_Ty', 'pose_Tz']])  # 5. min_max_scaler
        df[['pose_Tx', 'pose_Ty', 'pose_Tz']] = min_max_scaler.transform(df[['pose_Tx', 'pose_Ty', 'pose_Tz']])
        # print(df)

        print('>>>>>> `medium`', [i for i in zip(list(df.medium.value_counts().index),
                                                 list(df.medium.value_counts().values))])  # 6. medium, pid -> interaction (AV * 1)
        print('>>>>>> `pid`', [i for i in zip(list(df.pid.value_counts().index),
                                              list(df.pid.value_counts().values))])
        if len(list(set(list(df.medium)))) == 1 and 'AV' in list(set(list(df.medium))):
            df['AV'] = 1
            df['FTF'] = 0
        elif len(list(set(list(df.medium)))) == 1 and 'FTF' in list(set(list(df.medium))):
            df['FTF'] = 1
            df['AV'] = 0
        else:
            df = pd.concat([df, pd.get_dummies(df.medium)], axis=1, join='inner')
        # print(df)

        if len(list(set(list(df.pid)))) == 1 and '1' in list(set(list(df.pid))):
            df['1'] = 1
            df['2'] = 0
        elif len(list(set(list(df.pid)))) == 1 and '2' in list(set(list(df.pid))):
            df['2'] = 1
            df['1'] = 0
        else:
            df = pd.concat([df, pd.get_dummies(df.pid)], axis=1, join='inner')
        # print(df)

        df['interaction'] = df['AV'] * df['1']
        print('>>>>>> `interaction`', [i for i in zip(list(df.interaction.value_counts().index),
                                                      list(df.interaction.value_counts().values))])

        df = df.set_index(['pcode'])  # 7. set index pcode -> _gm & _gmc
        for col_name in df.loc[:, 'gaze_0_x': 'pose_Rz'].columns:
            for i in list(set(df.index)):
                df.loc[i, col_name + '_gm'] = df.groupby(by='pcode')[col_name].mean()[i]

        for col_name in df.loc[:, 'gaze_0_x': 'pose_Rz'].columns:
            df[col_name + '_gmc'] = df[col_name] - df[col_name + '_gm']
        # print(df)

        df['pid_0'] = df['1']
        df['medium_1'] = df['AV']
        # print(df)

        df = df.drop(columns=['scode', 'confidence', 'success', 'gaze_0_x', 'gaze_0_y',
                              'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x',
                              'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx',
                              'pose_Ry', 'pose_Rz', 'coded', 'medium', 'pid', 'AV', 'FTF', '1', '2'], axis=1)

        if len(list(set(list(df.tag)))) == 1 and '1' in list(set(list(df.tag))):  # 8. tag dummy
            df['1'] = 1
            df['2'] = 0
        elif len(list(set(list(df.tag)))) == 1 and '2' in list(set(list(df.tag))):
            df['2'] = 1
            df['1'] = 0
        else:
            df = pd.concat([df, pd.get_dummies(df.tag)], axis=1, join='inner')
        df['tag_1'] = df['1']

        df = df.drop(columns=['tag', '1', '2'], axis=1)
        # print(df)

        # watch out for the reference pcode
        if '04oia83a' in df.columns:
            df.drop('04oia83a', axis=1, inplace=True)
        else:
            df = df

        print('>>>>>> Final data has %s rows and %s columns (164 columns is right)' % (df.shape[0], df.shape[1]))
        return df

    def apply(self, uncoded_data_file):
        new_data = self.main(uncoded_data_file)
        new_data = new_data.apply(pd.to_numeric, errors='ignore')
        model = joblib.load('/'.join(self.column_names_file.split('/')[:-1]) + '/model.pkl')
        new_data['result'] = model.predict(new_data.values[:, :].astype(float))
        return new_data

if __name__ == '__main__':
    new_predict = predict(args.fp)
    for file in os.listdir(new_predict.uncoded_data_folder):
        if '.DS_Store' not in file:  # this line is essential on the OS system
            final_data = new_predict.apply(new_predict.uncoded_data_folder + '/' + file)
            final_data.to_csv('/'.join(new_predict.column_names_file.split('/')[:-2]) +
                              '/uncoded_data_result/' +
                              str(file).split('.csv')[0] +
                              '_result.csv',
                              sep=',', encoding='utf-8', header=True, index=None)
            print('\n')
    print('Finished :)')
