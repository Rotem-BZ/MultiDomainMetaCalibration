import json
from collections import defaultdict
import copy
import numpy as np
from matplotlib import pyplot as plt


# following the ARM paper, we will set aside the following as the test set:
# ['Impulse Noise', 'Motion Blur', 'Fog', 'Elastic'] - all severity levels
# ['Spatter', 'JPEG'] - severity 5

def dict_print(d):
    d = {k: round(v, 3) for k, v in d.items()}
    print(d)

def analyze(res):
    test_perturbations = ['impulse_noise',
                          'motion_blur', 'fog',
                          'elastic_transform', 'jpeg_compression_5',
                          'spatter_5']
    test_dict = defaultdict(list)
    train_dict = defaultdict(list)
    final_test_dict = {}
    final_train_dict = {}
    best_epoch = np.argmax(res['val_acc'])
    test_set_acc = round(res['test_acc'][best_epoch], 3)
    test_set_ece = round(res['test_ece'][best_epoch], 4)
    print(f'standard test set err {1-test_set_acc} ece {test_set_ece} time {round(res["time"][0]/3600, 3)}')
    res = {k: v[0] for k, v in res.items() if len(v) == 1 and isinstance(v[0], dict)}
    for pert, pert_data in res.items():
        if any(test_pert in pert for test_pert in test_perturbations):
            for k, v in pert_data.items():
                test_dict[k].append(v)
        else:
            for k, v in pert_data.items():
                train_dict[k].append(v)
    metrics = ['acc', 'err', 'ece']
    for k, v in test_dict.items():
        if any(f'test_{met}' == k for met in metrics):
            final_test_dict[f'test-doms_high_{k}'] = max(v)
            # final_test_dict[f'test-doms_low_{k}'] = min(v)
            final_test_dict[f'test-doms_avg_{k}'] = sum(v) / len(v)
            # print(len(v))
    dict_print(final_test_dict)
    for k, v in train_dict.items():
        if any(f'test_{met}' == k for met in metrics):
            final_train_dict[f'train-doms_high_{k}'] = max(v)
            # final_train_dict[f'train-doms_low_{k}'] = min(v)
            final_train_dict[f'train-doms_avg_{k}'] = sum(v) / len(v)
            # print(len(v))
    dict_print(final_train_dict)
    return final_test_dict, final_train_dict


# def update_json_experiment_log_dict(json_experiment_log_file_name):
#     with open(json_experiment_log_file_name, 'r') as f:
#         summary_dict = json.load(fp=f)
#     copy_dict = copy.deepcopy(summary_dict)
#     for key, v in summary_dict.items():
#         if len(v) == 1 and isinstance(v[0], dict):
#             del copy_dict[key]
#
#     with open(json_experiment_log_file_name, 'w') as f:
#         json.dump(copy_dict, fp=f)


if __name__ == '__main__':
    experiments = ['resnet18_cross_entropy_350', 'md_meta', 'multi_meta', 'md_train_meta', 'md_multi_meta', 'md_train', 'no_meta_350', 'md_train_no_meta']
    for exper in experiments:
        print(exper)
        with open(f'/home/niv.ko/MetaCalibration/Experiments/{exper}.json', 'r') as f:
            exper_dict = json.load(fp=f)
        analyze(exper_dict)

