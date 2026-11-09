'''
Concrete ResultModule class for saving experiment results
'''

import os
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'wb')
        pickle.dump(self.data, f)
        f.close()
