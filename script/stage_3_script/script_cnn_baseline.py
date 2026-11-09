'''
Train the baseline CNN for one dataset, print the test results and save its learning curves.
usage: python script_cnn_baseline.py [mnist|orl|cifar]
'''

import sys
import numpy as np
import torch
from configs import BASELINES, run_experiment


def main(dataset_key):
    # fixed seeds -> repeatable results
    np.random.seed(2)
    torch.manual_seed(2)

    print('************ Start ************')
    metrics, result = run_experiment(dataset_key, BASELINES[dataset_key], 'baseline')
    print('************ Overall Performance ************')
    print('%s CNN | accuracy: %.4f  precision: %.4f  recall: %.4f  f1: %.4f  (train time %.1fs)' % (
        dataset_key.upper(), metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1'], result['train_time']))
    print('************ Finish ************')


if __name__ == '__main__':
    main(sys.argv[1].lower() if len(sys.argv) > 1 else 'mnist')
