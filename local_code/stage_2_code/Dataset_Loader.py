'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    train_file_name = None
    test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_name):
        X = []
        y = []
        f = open(self.dataset_source_folder_path + file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])    # label is first, features follow
            y.append(elements[0])
        f.close()
        return X, y

    def load(self):
        print('loading data...')
        X_train, y_train = self.load_file(self.train_file_name)
        X_test, y_test = self.load_file(self.test_file_name)
        return {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}}