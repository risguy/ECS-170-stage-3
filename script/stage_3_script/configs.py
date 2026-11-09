'''
Shared experiment definitions for the stage 3 scripts: dataset locations, baseline CNN
configurations, the configuration study grid, and a helper that runs one experiment end to end.
'''

import copy
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Pre_Split import Setting_Pre_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Learning_Curve_Plotter import Learning_Curve_Plotter

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data') + os.sep
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result') + os.sep

DATASET_FILES = {'mnist': 'MNIST', 'orl': 'ORL', 'cifar': 'CIFAR'}


def conv_stack(channels, kernel_size=3, padding=1, stride=1, pool='max', pool_size=2,
               pool_stride=2, batch_norm=False):
    """Build the `conv_layers` list: one block per entry of `channels`."""
    return [{'out_channels': c, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding,
             'pool': pool, 'pool_size': pool_size, 'pool_stride': pool_stride,
             'batch_norm': batch_norm} for c in channels]


# ------------------------------------------------------------------ baselines
BASELINES = {
    # 28x28x1, 60k train images
    'mnist': {'conv_layers': conv_stack([32, 64]), 'hidden_dims': [128],
              'dropout': 0.25, 'epochs': 10, 'batch_size': 128, 'learning_rate': 1e-3},
    # 112x92x1, only 360 train images for 40 people -> small net, dropout, small batches
    'orl': {'conv_layers': conv_stack([16, 32]), 'hidden_dims': [128],
            'dropout': 0.3, 'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-3},
    # 32x32x3, 50k train images -> deeper net with batch norm
    'cifar': {'conv_layers': conv_stack([32, 64, 128], batch_norm=True), 'hidden_dims': [256],
              'dropout': 0.3, 'epochs': 20, 'batch_size': 128, 'learning_rate': 1e-3},
}

# the configuration study uses fewer epochs so the whole grid finishes in reasonable time
STUDY_EPOCHS = {'mnist': 5, 'orl': 20, 'cifar': 10}


def study_grid(dataset_key):
    """One-factor-at-a-time variations of the baseline.
    Returns {factor: {label: config}}; every config is the baseline with a single change."""
    base = copy.deepcopy(BASELINES[dataset_key])
    base['epochs'] = STUDY_EPOCHS[dataset_key]
    bn = base['conv_layers'][0]['batch_norm']
    base_channels = [l['out_channels'] for l in base['conv_layers']]

    def variant(**changes):
        cfg = copy.deepcopy(base)
        cfg.update(changes)
        return cfg

    all_channels = [16, 32, 64] if dataset_key == 'orl' else [32, 64, 128]
    grid = {
        'depth': {
            '1 conv layer': variant(conv_layers=conv_stack(all_channels[:1], batch_norm=bn)),
            '2 conv layers': variant(conv_layers=conv_stack(all_channels[:2], batch_norm=bn)),
            '3 conv layers': variant(conv_layers=conv_stack(all_channels[:3], batch_norm=bn)),
        },
        'kernel size': {
            'kernel 3': variant(conv_layers=conv_stack(base_channels, kernel_size=3, padding=1, batch_norm=bn)),
            'kernel 5': variant(conv_layers=conv_stack(base_channels, kernel_size=5, padding=2, batch_norm=bn)),
            'kernel 7': variant(conv_layers=conv_stack(base_channels, kernel_size=7, padding=3, batch_norm=bn)),
        },
        'padding': {
            'padding 0 (valid)': variant(conv_layers=conv_stack(base_channels, padding=0, batch_norm=bn)),
            'padding 1 (same)': variant(conv_layers=conv_stack(base_channels, padding=1, batch_norm=bn)),
        },
        'stride': {
            'conv stride 1': variant(conv_layers=conv_stack(base_channels, stride=1, batch_norm=bn)),
            'conv stride 2': variant(conv_layers=conv_stack(base_channels, stride=2, batch_norm=bn)),
        },
        'pooling': {
            'max pool': variant(conv_layers=conv_stack(base_channels, pool='max', batch_norm=bn)),
            'avg pool': variant(conv_layers=conv_stack(base_channels, pool='avg', batch_norm=bn)),
            'no pool': variant(conv_layers=conv_stack(base_channels, pool=None, batch_norm=bn)),
        },
        'hidden dim': {
            'hidden 32': variant(hidden_dims=[32]),
            'hidden 128': variant(hidden_dims=[128]),
            'hidden 512': variant(hidden_dims=[512]),
            'hidden 256-128': variant(hidden_dims=[256, 128]),
        },
        'loss function': {
            'cross entropy': variant(loss='cross_entropy'),
            'label smoothing CE': variant(loss='label_smoothing'),
            'MSE (softmax vs one-hot)': variant(loss='mse'),
        },
        'activation': {
            'relu': variant(activation='relu'),
            'tanh': variant(activation='tanh'),
            'leaky relu': variant(activation='leaky_relu'),
        },
    }
    return grid


# ------------------------------------------------------------------ runner
def make_dataset(dataset_key):
    data_obj = Dataset_Loader(dataset_key, '')
    data_obj.dataset_source_folder_path = DATA_FOLDER
    data_obj.dataset_source_file_name = DATASET_FILES[dataset_key]
    return data_obj


def run_experiment(dataset_key, config, run_name, data_obj=None, plot=True):
    """Train/evaluate one CNN configuration. Returns (metrics, saved result dict)."""
    if data_obj is None:
        data_obj = make_dataset(dataset_key)

    method_obj = Method_CNN('CNN-' + run_name, '', config)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER + dataset_key + os.sep
    result_obj.result_destination_file_name = run_name.replace(' ', '_').replace('/', '-') + '.pkl'

    setting_obj = Setting_Pre_Split('pre-split', '')
    evaluate_obj = Evaluate_Metrics('metrics', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()

    result = result_obj.data
    if plot:
        Learning_Curve_Plotter.plot(
            result['history'], '%s - %s (test acc %.4f)' % (dataset_key.upper(), run_name, metrics['accuracy']),
            RESULT_FOLDER + dataset_key + os.sep + 'curve_' + run_name.replace(' ', '_').replace('/', '-') + '.png')
    return metrics, result
