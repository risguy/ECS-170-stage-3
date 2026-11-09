'''
Concrete IO class for the stage 3 image datasets (MNIST, ORL, CIFAR-10)
'''

import pickle
import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # ORL images are stored as 112x92x3 with three identical channels.
    # By default only the first channel is kept; set to True to keep all three.
    use_all_channels = False
    # standardize every channel with the mean/std computed on the training set
    normalize = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    @staticmethod
    def _to_array(instances, use_all_channels):
        images = []
        for instance in instances:
            img = np.asarray(instance['image'], dtype=np.float32)
            if img.ndim == 2:
                # gray-scale (H, W) -> (1, H, W)
                img = img[np.newaxis, :, :]
            else:
                # colour (H, W, C) -> (C, H, W), the layout PyTorch expects
                img = np.transpose(img, (2, 0, 1))
            images.append(img)
        X = np.stack(images)
        if not use_all_channels and X.shape[1] == 3 and np.allclose(X[:, 0], X[:, 1]) and np.allclose(X[:, 0], X[:, 2]):
            # the three channels are identical (ORL), so a single one is enough
            X = X[:, :1]
        return X

    def load(self):
        # the datasets are large, so keep them in memory when load() is called repeatedly
        if self.data is not None:
            return self.data
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        raw = pickle.load(f)
        f.close()

        X_train = self._to_array(raw['train'], self.use_all_channels)
        X_test = self._to_array(raw['test'], self.use_all_channels)
        y_train = np.array([inst['label'] for inst in raw['train']], dtype=np.int64)
        y_test = np.array([inst['label'] for inst in raw['test']], dtype=np.int64)

        # labels must start at 0 for the loss functions (ORL labels are 1..40)
        label_offset = int(y_train.min())
        y_train -= label_offset
        y_test -= label_offset

        # scale pixels to [0, 1], then standardize per channel with training statistics
        X_train /= 255.0
        X_test /= 255.0
        if self.normalize:
            # statistics in float64: float32 accumulation over ~50M values is not accurate enough
            mean = X_train.mean(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
            std = X_train.std(axis=(0, 2, 3), keepdims=True, dtype=np.float64) + 1e-8
            mean, std = mean.astype(np.float32), std.astype(np.float32)
            for arr in (X_train, X_test):   # in place, to avoid extra copies of the large arrays
                arr -= mean
                arr /= std

        print('train:', X_train.shape, 'test:', X_test.shape,
              'classes:', int(max(y_train.max(), y_test.max())) + 1)
        self.data = {'train': {'X': X_train, 'y': y_train},
                     'test': {'X': X_test, 'y': y_test}}
        return self.data
