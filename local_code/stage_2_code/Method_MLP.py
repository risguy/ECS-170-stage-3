'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 128

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(128, 10)   # 10 classes, raw logits

    def forward(self, x):
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        return self.fc_layer_3(h)

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X = torch.FloatTensor(np.array(X)) / 255.0   # remove "/ 255.0" if your loader already scales
        y = torch.LongTensor(np.array(y))
        n = X.shape[0]

        for epoch in range(self.max_epoch):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                y_pred = self.forward(X[idx])
                train_loss = loss_function(y_pred, y[idx])

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * len(idx)

            with torch.no_grad():
                train_pred = self.forward(X).max(1)[1]
            accuracy_evaluator.data = {'true_y': y, 'pred_y': train_pred}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(),
                  'Loss:', epoch_loss / n)

    def test(self, X):
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)) / 255.0)   # same scaling as train
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}