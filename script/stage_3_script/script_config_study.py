'''
Configuration study: change one CNN setting at a time (depth, kernel size, padding, stride,
pooling, hidden dimension, loss function, activation) and report the effect on test performance.
usage: python script_config_study.py [mnist|orl|cifar|all]
'''

import csv
import os
import sys
from configs import RESULT_FOLDER, make_dataset, run_experiment, study_grid
from local_code.stage_3_code.Learning_Curve_Plotter import Learning_Curve_Plotter


def study(dataset_key):
    out_dir = RESULT_FOLDER + dataset_key + os.sep
    data_obj = make_dataset(dataset_key)   # load the (large) dataset only once
    rows = []
    for factor, variants in study_grid(dataset_key).items():
        histories = {}
        for label, config in variants.items():
            print('\n======== %s | %s: %s ========' % (dataset_key.upper(), factor, label))
            metrics, result = run_experiment(dataset_key, config, 'study_%s_%s' % (factor, label),
                                             data_obj=data_obj, plot=False)
            histories[label] = result['history']
            rows.append({'dataset': dataset_key, 'factor': factor, 'setting': label,
                         'test_acc': round(metrics['accuracy'], 4), 'f1': round(metrics['f1'], 4),
                         'final_train_acc': round(result['history']['train_acc'][-1], 4),
                         'final_train_loss': round(result['history']['train_loss'][-1], 4),
                         'params': result['history']['num_params'],
                         'train_time_s': round(result['train_time'], 1)})
        Learning_Curve_Plotter.plot_comparison(
            histories, '%s - effect of %s' % (dataset_key.upper(), factor),
            out_dir + 'study_' + factor.replace(' ', '_') + '.png')

    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + 'study_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print('\n************ %s configuration study ************' % dataset_key.upper())
    print('%-14s %-28s %8s %8s %10s %10s' % ('factor', 'setting', 'test acc', 'F1', 'params', 'time(s)'))
    for r in rows:
        print('%-14s %-28s %8.4f %8.4f %10d %10.1f' % (
            r['factor'], r['setting'], r['test_acc'], r['f1'], r['params'], r['train_time_s']))


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    for key in (['mnist', 'orl', 'cifar'] if arg == 'all' else [arg]):
        study(key)
