# env: Python 3.7.4
# You can run this script in server (scribe in UCD)
# Lyndon on Feb. 11, 2020
# Revised on Feb. 25, 2020
import joblib
import argparse
import warnings
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

parser = argparse.ArgumentParser()
parser.add_argument('-fp', type=str, nargs='+', help='Parameter: the absolute path of the coded data')
args = parser.parse_args()


class model:
    def __init__(self, file_path):
        self.coded_data = file_path[0]

    def preprocessing(self):
        # pid needs to be string, which is more suitable for the dummy variable generation
        data = pd.read_csv(self.coded_data, sep=',', header=0, dtype={'pid': str})
        print('>>>>>> Original data for training has %s rows and %s columns'%(data.shape[0], data.shape[1]))
        print('>>>>>> Number of `pcode` is: %s' % (len(set(data.pcode))))
        data.drop('scode.1', axis=1, inplace=True)

        data = pd.concat([data, pd.get_dummies(data.pcode)], axis=1, join='inner')

        data.replace({'coder_result': {0: 'O', 1: 'S', 2: 'T'}}, inplace=True)
        print('>>>>>> `coder_result`', [i for i in zip(list(data.coder_result.value_counts().index),
                                                     list(data.coder_result.value_counts().values))])

        data.success = data.success.replace([0, 1], ['FALSE', 'TRUE'])
        print('>>>>>> `success`', [i for i in zip(list(data.success.value_counts().index),
                                                  list(data.success.value_counts().values))])

        data = data[data['success'] != 'FALSE']
        print('>>>>>> Now data has %s rows and %s columns' %(data.shape[0], data.shape[1]))

        data = data[data['confidence'] >= .85]
        print('>>>>>> Now data has %s rows and %s columns' %(data.shape[0], data.shape[1]))

        data.tag = data.tag.apply(lambda x: str(x.split('_')[1].strip()))
        print('>>>>>> `tag`', [i for i in zip(list(data.tag.value_counts().index),
                                              list(data.tag.value_counts().values))])

        temp = data.pop('coder_result')
        data.insert(0, 'coder_result', temp)

        min_max_scaler.fit(data[['pose_Tx', 'pose_Ty', 'pose_Tz']])
        data[['pose_Tx', 'pose_Ty', 'pose_Tz']] = min_max_scaler.transform(data[['pose_Tx', 'pose_Ty', 'pose_Tz']])

        print('>>>>>> `medium`', [i for i in zip(list(data.medium.value_counts().index),
                                                 list(data.medium.value_counts().values))])
        print('>>>>>> `pid`', [i for i in zip(list(data.pid.value_counts().index),
                                              list(data.pid.value_counts().values))])
        data = pd.concat([data, pd.get_dummies(data.medium)], axis=1, join='inner')
        data = pd.concat([data, pd.get_dummies(data.pid)], axis=1, join='inner')
        data['interaction'] = data['AV'] * data['1']
        print('>>>>>> interaction', [i for i in zip(list(data.interaction.value_counts().index),
                                                    list(data.interaction.value_counts().values))])

        data = data.set_index(['pcode'])
        for col_name in data.loc[:, 'gaze_0_x': 'pose_Rz'].columns:
            for i in list(set(data.index)):
                data.loc[i, col_name + '_gm'] = data.groupby(by='pcode')[col_name].mean()[i]
        for col_name in data.loc[:, 'gaze_0_x': 'pose_Rz'].columns:
            data[col_name + '_gmc'] = data[col_name] - data[col_name + '_gm']

        # pid_0 ('1') * medium_1 ('AV')
        data = data.rename(columns={'1': 'pid_0', '2': 'pid_1', 'AV': 'medium_1', 'FTF': 'medium_0'})
        data = data.drop(columns=['scode', 'confidence', 'success', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                  'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
                                  'pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz',
                                  'coded', 'medium', 'pid', 'pid_1', 'medium_0', '04oia83a'], axis=1)
        data = pd.concat([data, pd.get_dummies(data.tag)], axis=1, join='inner')
        data.drop('tag', axis=1, inplace=True)
        data = data.iloc[:, :-1]
        data.rename(columns={'1': 'tag_1'}, inplace=True)
        print('>>>>>> Final data has: %s rows and %s columns' %(data.shape[0], data.shape[1]))
        return data

    def training(self, material):
        data = material.apply(pd.to_numeric, errors='ignore')
        array = data.values
        X = array[:, 1:].astype(float)
        Y = array[:, 0]
        validation_size = .2
        seed = 2468
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                                                        test_size=validation_size,
                                                                        random_state=seed)
        print('>>>>>> %s size: %s, %s size: %s' %('Training set\'s', len(X_train), 'Validation set\'s', len(X_validation)))
        model = MLPClassifier(activation='tanh', solver='adam')
        model.fit(X = X_train, y = Y_train)
        predictions = model.predict(X_validation)
        print('>>>>>> Accuracy_score of the ML model:\n', accuracy_score(Y_validation, predictions))
        print('>>>>>> Classification report of the ML model:\n', classification_report(Y_validation, predictions))
        joblib.dump(model, '/'.join(self.coded_data.split('/')[:-1]) + '/model.pkl')
        joblib.dump(model, '/'.join(self.coded_data.split('/')[:-1]) + '/model.m')
        print('>>>>>> Finished :)')

if __name__ == '__main__':
    model = model(args.fp)
    ML_material = model.preprocessing()
    with open('/'.join(model.coded_data.split('/')[:-1]) + '/column_names.txt', 'w', encoding='utf-8') as t:
        for name in ML_material.columns:
            t.write(name.strip() + '\n')
    model.training(ML_material)
