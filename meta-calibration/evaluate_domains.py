import json
import os
import sys
import numpy as np
import torch
import time
import random
import argparse
from torch import nn
from torchvision import transforms
import torch.backends.cudnn as cudnn

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100

# Import network architectures
from Net.resnet import resnet18, resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature


# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100
}

# Mapping model name to model function
models = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar
}


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet18'
    save_loc = './'
    saved_model_name = 'resnet18_cross_entropy_350.model'
    exp_name = 'resnet18_cross_entropy'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--exp_name", type=str, default=exp_name,
                        dest="exp_name", help="name of the experiment")
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")

    return parser.parse_args()


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


def update_json_experiment_log_dict(experiment_update_dict, json_experiment_log_file_name):
    with open(json_experiment_log_file_name, 'r') as f:
        summary_dict = json.load(fp=f)

    for key in experiment_update_dict:
        if key not in summary_dict:
            summary_dict[key] = []
        summary_dict[key].append(experiment_update_dict[key])

    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(summary_dict, fp=f)


def get_test_loader_domain(dataset_images,
                           dataset_labels,
                           severity, batch_size,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    dataset_images = dataset_images[start_index:end_index]
    dataset_labels = dataset_labels[start_index:end_index]
    # include normalization
    dataset = torch.utils.data.TensorDataset(
        normalize(dataset_images), dataset_labels)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    saved_model_name = args.saved_model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]

    # dataset-root will be /Data/CIFAR-10-C
    _, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        random_seed=1,
        pin_memory=args.gpu
    )
    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0)
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    net.load_state_dict(torch.load(
        args.save_loc + args.saved_model_name))

    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    scaled_model = ModelWithTemperature(net, args.log)
    scaled_model.set_temperature(
        val_loader, cross_validate=cross_validation_error)
    T_opt = scaled_model.get_temperature()

    dataset_labels = torch.from_numpy(
        np.load(os.path.join(args.dataset_root, 'labels.npy'))).long()
    perturbations = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                     'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                     'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                     'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    start_time = time.time()
    for perturbation in perturbations:
        print('Current perturbation: ' + perturbation)
        for severity in range(1, 6):
            # load the data
            dataset_images = torch.from_numpy(np.float32(np.load(os.path.join(
                args.dataset_root, perturbation + '.npy')).transpose((0, 3, 1, 2)))) / 255.

            test_loader = get_test_loader_domain(
                dataset_images,
                dataset_labels,
                severity,
                batch_size=args.test_batch_size,
                pin_memory=args.gpu
            )

            logits, labels = get_logits_labels(test_loader, net)
            conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(
                logits, labels)

            p_ece = ece_criterion(logits, labels).item()
            p_adaece = adaece_criterion(logits, labels).item()
            p_cece = cece_criterion(logits, labels).item()
            p_nll = nll_criterion(logits, labels).item()

            logits, labels = get_logits_labels(test_loader, scaled_model)
            conf_matrix, accuracy, _, _, _ = test_classification_net_logits(
                logits, labels)

            ece = ece_criterion(logits, labels).item()
            adaece = adaece_criterion(logits, labels).item()
            cece = cece_criterion(logits, labels).item()
            nll = nll_criterion(logits, labels).item()

            # save the statistics
            experiment_update_dict = {perturbation + '_' + str(severity): {
                'test_err': 1 - p_accuracy,
                'test_nll': p_nll,
                'test_ece': p_ece,
                'test_aece': p_adaece,
                'test_cece': p_cece,
                'test_err_t': 1 - accuracy,
                'test_nll_t': nll,
                'test_ece_t': ece,
                'test_aece_t': adaece,
                'test_cece_t': cece,
                'opt_t': T_opt,
            }}
            json_experiment_log_file_name = os.path.join(
                'Experiments', args.exp_name + '.json')

            update_json_experiment_log_dict(
                experiment_update_dict, json_experiment_log_file_name)

    experiment_update_dict = {
        'domain_eval_time': time.time() - start_time
    }
    
    json_experiment_log_file_name = os.path.join(
        'Experiments', args.exp_name + '.json')

    update_json_experiment_log_dict(
        experiment_update_dict, json_experiment_log_file_name)
