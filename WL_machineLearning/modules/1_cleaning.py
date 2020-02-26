# env: Python 2.7.13
# Run this script locally, not on server
# Lyndon on Feb. 8, 2020
# Revised on Feb. 25, 2020
import argparse
import graphlab as gl

parser = argparse.ArgumentParser()
parser.add_argument('-fp', type=str, nargs='+', help='The first parameter: absolute file path of the mixed dataset. '
                                                     '\nThe second parameter: absolute file path of the survey dataset')
args = parser.parse_args()


class cleaning:
    def __init__(self, paths):
        self.mainFile = paths[0]
        self.surveyFile = paths[1]

    def load_data(self):
        data = gl.SFrame.read_csv(self.mainFile, header=True, delimiter=',', column_type_hints={'section': int})
        print('>>>>>> Original mixed dataset has %s rows and %s columns' %(data.shape[0], data.shape[1]))
        print('>>>>>> In original dataset, Number of `pcode` is: %s, number of `scode` is: %s'
              %(len(set(list(data['pcode']))), len(set(list(data['scode'])))))
        data = data[data['success'].apply(lambda x: x >= 0 and x <= 1)]
        data = data[data['confidence'].apply(lambda x: x >= 0. and x <= 1.)]
        print('>>>>>> After filtering by `success` and `confidence`')
        print('>>>>>> Mixed dataset has %s rows and %s columns' %(data.shape[0], data.shape[1]))
        print('>>>>>> Number of `pcode` is: %s, number of `scode` is: %s'
              %(len(set(list(data['pcode']))), len(set(list(data['scode'])))))
        data = data.remove_columns(['section', 'section_start', 'section_end', 'timestamp', 'filename', 'frame',
                                    'time', 'face_id', 'coder1', 'coder2', 'coder3', 'coder4', 'coder5', 'coder6',
                                    'coder7', 'coder8', 'coder9', 'coder10', 'coder11', 'AU01_r', 'AU02_r', 'AU04_r',
                                    'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
                                    'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c',
                                    'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c',
                                    'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c',
                                    'utt_id', 'word', 'word_conf', 'speaker', 'speaker_conf', 'speaker_final'])
        return data

    def join_data(self):
        survey = gl.SFrame.read_csv(self.surveyFile, header=True, delimiter=',')
        survey = survey[['scode', 'pcode', 'medium', 'pid']]
        survey = survey.unique()
        final = self.load_data().join(survey, on='pcode', how='left')
        print('>>>>>> Final combined data has %s rows and %s columns' %(final.shape[0], final.shape[1]))
        return final

    def separate(self):
        final = self.join_data()
        final_coded = final[final['coded'] == 'TRUE']
        print('>>>>>> Final coded data has %s rows and %s columns' %(final_coded.shape[0], final_coded.shape[1]))
        final_coded.save('/'.join(self.mainFile.split('/')[:-1]) + '/coded.csv', format='csv')
        final_uncoded = final[final['coded'] == 'FALSE']
        print('>>>>>> Final uncoded data has %s rows and %s columns' %(final_uncoded.shape[0], final_uncoded.shape[1]))
        final_uncoded.save('/'.join(self.mainFile.split('/')[:-1]) + '/uncoded.csv', format='csv')
        print('>>>>>> Finished :)')

if __name__ == '__main__':
    cleaning = cleaning(args.fp)
    cleaning.separate()
