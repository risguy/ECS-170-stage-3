'''
Concrete MethodModule class for a configurable CNN image classifier.

One class serves MNIST, ORL and CIFAR-10: the architecture (depth, kernel size, padding,
stride, pooling, hidden dimensions, ...), the loss function and the optimizer are all
controlled through `config`, so a new experiment only needs a new config dictionary.
'''

import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from local_code.base_class.method import method


# default configuration; any key can be overridden by the config passed to Method_CNN
DEFAULT_CONFIG = {
    # --- architecture ---
    # one dict per convolutional block: conv -> (batch norm) -> activation -> (pooling)
    'conv_layers': [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'pool': 'max', 'pool_size': 2, 'pool_stride': 2, 'batch_norm': False},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1,
         'pool': 'max', 'pool_size': 2, 'pool_stride': 2, 'batch_norm': False},
    ],
    'hidden_dims': [128],        # fully connected hidden layers after flattening
    'activation': 'relu',        # relu | tanh | leaky_relu | sigmoid
    'dropout': 0.0,              # dropout probability before every FC layer
    # --- optimization ---
    'loss': 'cross_entropy',     # cross_entropy | label_smoothing | nll | mse
    'optimizer': 'adam',         # adam | sgd
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'momentum': 0.9,             # only used by sgd
    'epochs': 20,
    'batch_size': 128,
    'seed': 0,
}

ACTIVATIONS = {
    'relu': nn.ReLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU, 'sigmoid': nn.Sigmoid,
}


class CNN_Model(nn.Module):
    """The actual PyTorch network, built from the config."""

    def __init__(self, input_shape, num_classes, config):
        super().__init__()
        act = ACTIVATIONS[config['activation']]
        in_channels = input_shape[0]

        conv_blocks = []
        for spec in config['conv_layers']:
            conv_blocks.append(nn.Conv2d(in_channels, spec['out_channels'],
                                         kernel_size=spec.get('kernel_size', 3),
                                         stride=spec.get('stride', 1),
                                         padding=spec.get('padding', 0)))
            if spec.get('batch_norm', False):
                conv_blocks.append(nn.BatchNorm2d(spec['out_channels']))
            conv_blocks.append(act())
            pool = spec.get('pool')
            if pool == 'max':
                conv_blocks.append(nn.MaxPool2d(spec.get('pool_size', 2), spec.get('pool_stride', None)))
            elif pool == 'avg':
                conv_blocks.append(nn.AvgPool2d(spec.get('pool_size', 2), spec.get('pool_stride', None)))
            in_channels = spec['out_channels']
        self.conv = nn.Sequential(*conv_blocks)

        # work out the flattened feature size with a dummy forward pass
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, *input_shape)).numel()

        fc_layers = []
        in_dim = flat_dim
        for hidden in config['hidden_dims']:
            if config['dropout'] > 0:
                fc_layers.append(nn.Dropout(config['dropout']))
            fc_layers.append(nn.Linear(in_dim, hidden))
            fc_layers.append(act())
            in_dim = hidden
        if config['dropout'] > 0:
            fc_layers.append(nn.Dropout(config['dropout']))
        fc_layers.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc(h)  # raw class scores (logits)


class Method_CNN(method):
    data = None
    config = None
    model = None
    history = None

    def __init__(self, mName, mDescription, config=None):
        super().__init__(mName, mDescription)
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    # ------------------------------------------------------------------ helpers
    def _loss(self, logits, y):
        name = self.config['loss']
        if name == 'cross_entropy':
            return F.cross_entropy(logits, y)
        if name == 'label_smoothing':
            return F.cross_entropy(logits, y, label_smoothing=0.1)
        if name == 'nll':
            return F.nll_loss(F.log_softmax(logits, dim=1), y)
        if name == 'mse':
            # squared error between the softmax probabilities and the one-hot labels
            return F.mse_loss(F.softmax(logits, dim=1), F.one_hot(y, logits.shape[1]).float())
        raise ValueError('unknown loss: ' + name)

    def _optimizer(self):
        c = self.config
        if c['optimizer'] == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        if c['optimizer'] == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=c['learning_rate'],
                                   momentum=c['momentum'], weight_decay=c['weight_decay'])
        raise ValueError('unknown optimizer: ' + c['optimizer'])

    def _evaluate(self, X, y):
        """Loss, accuracy and predictions on a dataset (no gradient, eval mode)."""
        self.model.eval()
        total_loss, preds = 0.0, []
        bs = max(self.config['batch_size'], 256)
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = X[i:i + bs].to(self.device)
                yb = y[i:i + bs].to(self.device)
                logits = self.model(xb)
                total_loss += self._loss(logits, yb).item() * len(xb)
                preds.append(logits.argmax(1).cpu())
        preds = torch.cat(preds)
        return total_loss / len(X), (preds == y).float().mean().item(), preds

    # ----------------------------------------------------------------- training
    def train(self, X, y, X_test=None, y_test=None):
        c = self.config
        torch.manual_seed(c['seed'])
        np.random.seed(c['seed'])

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        X_test = torch.as_tensor(X_test, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, dtype=torch.long)

        num_classes = int(max(y.max(), y_test.max())) + 1
        self.model = CNN_Model(tuple(X.shape[1:]), num_classes, c).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print('model parameters:', n_params)
        optimizer = self._optimizer()

        self.history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
                        'num_params': n_params}
        generator = torch.Generator().manual_seed(c['seed'])

        for epoch in range(c['epochs']):
            self.model.train()
            perm = torch.randperm(len(X), generator=generator)
            for i in range(0, len(X), c['batch_size']):
                idx = perm[i:i + c['batch_size']]
                xb, yb = X[idx].to(self.device), y[idx].to(self.device)
                loss = self._loss(self.model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # learning-curve bookkeeping: full-pass evaluation in eval mode after every epoch
            tr_loss, tr_acc, _ = self._evaluate(X, y)
            te_loss, te_acc, _ = self._evaluate(X_test, y_test)
            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['test_loss'].append(te_loss)
            self.history['test_acc'].append(te_acc)
            print('Epoch: %d  train loss: %.4f  train acc: %.4f  test loss: %.4f  test acc: %.4f'
                  % (epoch + 1, tr_loss, tr_acc, te_loss, te_acc))

    def test(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), 256):
                preds.append(self.model(X[i:i + 256].to(self.device)).argmax(1).cpu())
        return torch.cat(preds)

    def run(self):
        print('method running...')
        print('--start training...')
        start = time.time()
        self.train(self.data['train']['X'], self.data['train']['y'],
                   self.data['test']['X'], self.data['test']['y'])
        train_time = time.time() - start
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y.numpy(), 'true_y': np.asarray(self.data['test']['y']),
                'history': self.history, 'config': self.config, 'train_time': train_time}
