'''
Concrete ResultModule class for loading saved experiment results
'''

import pickle

from local_code.base_class.result import result


class Result_Loader(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def load(self):
        print('loading results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'rb')
        self.data = pickle.load(f)
        f.close()
