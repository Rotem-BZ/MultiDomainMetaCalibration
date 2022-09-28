from typing import List
from math import floor

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.models import resnet18
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import networks
from Losses.loss import calculate_loss, ECELoss
from datasets import ColoredMNIST, RotatedMNIST, MultipleEnvironmentMNIST

import wandb
# wandb.init(project="meta-calibration")



class MetaCalibModule(pl.LightningModule):
    def __init__(self, model_type: str, input_shape, num_classes, num_domains, num_bins=15, t_a=100.0, t_b=0.01, **kws):
        # kws are used for the loss function parameters. num_bins, t_a and t_b are DECE parameters.
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        featurizer = networks.Featurizer(input_shape, self.hparams)
        classifier = networks.Classifier(featurizer.n_outputs, num_classes,
                                              self.hparams['nonlinear_classifier'], fast=True)
        self.network = nn.Sequential(featurizer, classifier)
        if model_type == 'md_meta':
            self.smooth_params = nn.ParameterList([nn.Parameter(torch.zeros(num_classes)) for _ in range(num_domains)])
        elif model_type == 'one_vec_meta':
            self.smooth_params = nn.ParameterList([nn.Parameter(torch.zeros(num_classes))]) # list of length 1
            # self.smooth_params = nn.Parameter(torch.zeros(num_classes))
        elif model_type == 'no_meta':
            self.smooth_params = torch.zeros(num_classes)
        else:
            raise ValueError(f"illegal model_type {model_type}")

        self.ece_criterion = ECELoss()

    def configure_optimizers(self):
        model_optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        if self.hparams.model_type == 'no_meta':
            return model_optimizer
        meta_optimizer = optim.Adam(self.smooth_params)
        return model_optimizer, meta_optimizer

    def forward(self, data):
        return self.network(data)

    def cross_entropy_with_smoothing(self, logits: torch.Tensor, domain_idx: int, labels: torch.Tensor):
        """
        perform cross entropy with label smoothing of a single domain.
        :param logits: model outputs for inputs from the given domain
        :param domain_idx: domain identifier
        :param labels: true classes
        :return: loss
        """
        if self.hparams.model_type == 'md_meta':
            smooth_params = self.smooth_params[domain_idx]  # this is a learnable vector
        elif self.hparams.model_type == 'one_vec_meta':
            smooth_params = self.smooth_params[0]
        else:
            smooth_params = self.smooth_params
        smooth_params = smooth_params.to(self.device)
        smooth_values = smooth_params[labels]   # inflate values to match every sample
        num_samples = labels.shape[0]
        soft_labels = torch.zeros((num_samples, self.hparams.num_classes), device=self.device)
        soft_labels[torch.arange(num_samples), labels] = 1.0 - smooth_values
        soft_labels = soft_labels + smooth_values.view(-1, 1) / self.hparams.num_classes

        # soft cross entropy. we use sum reduction.
        loss = torch.sum(torch.sum(- soft_labels * F.log_softmax(logits, dim=1), 1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.0, device=self.device)
        train_logits = []
        train_labels = []
        for domain_idx, domain_batch in enumerate(batch['train']):
            data, labels = domain_batch
            logits = self(data)
            train_logits.append(logits)
            train_labels.append(labels)
            loss += self.cross_entropy_with_smoothing(logits, domain_idx, labels)

        if self.hparams.model_type == 'no_meta':
            model_optimizer = self.optimizers()
            model_optimizer.zero_grad()
            self.manual_backward(loss)
            model_optimizer.step()
            meta_loss = 0.0
        else:
            meta_loss, loss = self.training_step_meta(batch, loss, train_logits, train_labels)

        train_logits = torch.vstack(train_logits).detach()
        train_labels = torch.cat(train_labels).detach()
        ece = self.ece_criterion(train_logits, train_labels).item()
        pred = torch.argmax(train_logits, dim=1)
        correct = pred == train_labels
        accuracy = correct.sum() / len(correct)

        self.log_dict({f'train_loss': loss,
                       f'train_ece': ece,
                       f'train_acc': accuracy,
                       f'meta_loss': meta_loss})

        return loss

    def training_step_meta(self, batch, network_loss, train_logits: list, train_labels: list):
        # batch is a dictionary 'train'-> (domain_batch_1, domain_batch_2, ...), 'meta'-> (domain_batch_1, domain_batch_2, ...)
        model_optimizer, meta_optimizer = self.optimizers()

        # fast_parameters = list(self.classifier.parameters())
        # for weight in self.network[1].parameters():
        #     weight.fast = None
        model_optimizer.zero_grad()
        grad = torch.autograd.grad(network_loss, self.network[1].parameters(), create_graph=True)
        fast_parameters = []
        current_lr = [param_group['lr'] for param_group in model_optimizer.param_groups][0]
        for k, weight in enumerate(self.network[1].parameters()):
            weight.fast = weight - current_lr * grad[k]
            fast_parameters.append(weight.fast)

        # outer loop
        # this will use the fast weights because they have not been reset to None
        logits_val_list = []
        labels_val_list = []
        for domain_batch in batch['meta']:
            data_val, labels_val = domain_batch
            logits = self(data_val)
            logits_val_list.append(logits)
            labels_val_list.append(labels_val)
        logits_val = torch.vstack(logits_val_list)
        labels_val = torch.cat(labels_val_list)

        meta_loss = calculate_loss(logits_val, labels_val, 'dece',
                                   num_bins=self.hparams.num_bins, t_a=self.hparams.t_a, t_b=self.hparams.t_b)
        meta_optimizer.zero_grad()
        self.manual_backward(meta_loss, retain_graph=True)
        meta_optimizer.step()

        # reset the fast weights to None
        for weight in self.network[1].parameters():
            weight.fast = None

        loss = torch.tensor(0.0, device=self.device)
        for domain_idx, (logits, labels) in enumerate(zip(train_logits, train_labels)):
            loss += self.cross_entropy_with_smoothing(logits, domain_idx, labels)

        model_optimizer.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 2)
        model_optimizer.step()

        return meta_loss.detach().cpu().item(), loss.detach()


    def validation_step(self, batch, batch_idx, *domain_idx):
        data, labels = batch
        logits = self(data)
        loss = calculate_loss(logits, labels, 'cross_entropy')

        ece = self.ece_criterion(logits, labels).item()

        pred = torch.argmax(logits, dim=1)
        correct = pred == labels
        accuracy = correct.sum() / len(correct)

        if len(domain_idx) > 0:
            domain_idx = domain_idx[0]
            self.log_dict({f'val_loss({self.hparams.model_type}_domain{domain_idx})': loss,
                           f'val_ece({self.hparams.model_type}_domain{domain_idx})': ece,
                           f'val_acc({self.hparams.model_type}_domain{domain_idx})': accuracy})
            return {'loss': loss, 'ece': ece, 'accuracy': accuracy}
        else:
            self.log_dict({f'val_loss({self.hparams.model_type})': loss,
                           f'val_ece({self.hparams.model_type})': ece,
                           f'val_acc({self.hparams.model_type})': accuracy})
            return {'loss': loss, 'ece': ece, 'accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        # if num_domains > 1, outputs is list (num domains) of lists (per batch) of dictionaries outputted by val step
        # if num_domains = 1, outputs is list (per batch) of dictionaries
        if self.hparams.num_domains == 1:
            outputs = [outputs]
        for domain_idx, domain_list in enumerate(outputs):
            for metric in domain_list[0].keys():
                values = [d[metric] for d in domain_list]
                self.log(f"avg_{metric}_domain{domain_idx}", sum(values) / len(values))


class DomainMNIST_DataModule(pl.LightningDataModule):
    train_size = 0.6
    meta_val_size = 0.1     # out of the train samples
    val_size = 0.1
    def __init__(self, data_dir: str = '', batch_size: int = 32, mnist_class=None, num_workers=5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mnist_class = mnist_class or ColoredMNIST
        self.num_workers = num_workers
        self.train_set = []
        self.meta_set = []
        self.val_set = []
        self.test_set = []
    def prepare_data(self):
        mnist_datasets = self.mnist_class('', None, dict()).datasets
        print(f"smallest domain size: {min(map(len, mnist_datasets))}")

        for dataset in mnist_datasets:
            data_len = len(dataset)
            train_len = floor(self.train_size * data_len)
            meta_val_len = floor(self.meta_val_size * train_len)
            val_len = floor(self.val_size * data_len)
            test_len = data_len - train_len - val_len

            train_set, meta_set, val_set, test_set =  random_split(dataset, [train_len - meta_val_len,
                                                                             meta_val_len,
                                                                             val_len,
                                                                             test_len])
            self.train_set.append(train_set)
            self.meta_set.append(meta_set)
            self.val_set.append(val_set)
            self.test_set.append(test_set)

            # fig, (ax1, ax2) = plt.subplots(2)

            # example_image_tensor, example_label = train_set[0]
            # example_image = Image.fromarray(example_image_tensor.numpy()[0])
            # plt.imshow(example_image)
            # plt.title(f"labels: {example_label.item()}")
            # plt.show()

            # ax1.imshow(example_image)
            # ax1.set_title(f"first. label: {example_label.item()}")
            # example_image = Image.fromarray(example_image_tensor.numpy()[1], mode="P")
            # ax2.imshow(example_image)
            # ax2.set_title(f"second. label: {example_label.item()}")
            # plt.show()
        print(f"\ntrain set size: {len(self.train_set[0])}\n")

    def train_dataloader(self):
        return {'train': [DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers) for dataset in self.train_set],
                'meta': [DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers) for dataset in self.meta_set]}
    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                           num_workers=self.num_workers) for dataset in self.val_set]
    # def test_dataloader(self):
    #     return [DataLoader(dataset, batch_size=self.batch_size, shuffle=True) for dataset in self.test_set]


def main():
    # debug mode operates on CPU without wandb and progress bar
    debug_mode = False
    model_type = ['no_meta', 'md_meta', 'one_vec_meta'][2]
    mnist_class_name = 'rotated'
    # mnist_class_name = 'colored'
    run_name = f"{model_type}_{mnist_class_name}"

    if mnist_class_name == 'rotated':
        input_shape = (1, 28, 28)
        num_classes = 10
        num_domains = 6
        # num_domains = 1
        mnist_class = RotatedMNIST
    elif mnist_class_name == 'colored':
        input_shape = (2, 28, 28)
        num_classes = 2
        num_domains = 3
        mnist_class = ColoredMNIST
    else:
        raise ValueError

    if not debug_mode:
        wandb.init(project="meta-calibration", name=run_name)
        wandb_logger = WandbLogger(name=run_name)
    else:
        wandb_logger = None
    model = MetaCalibModule(model_type, input_shape, num_classes=num_classes, num_domains=num_domains,
                            lr=1e-3, nonlinear_classifier=False, weight_decay=5e-5)
    datamodule = DomainMNIST_DataModule(batch_size=32, num_workers=4, mnist_class=mnist_class)
    if debug_mode:
        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
    else:
        # wandb_logger = WandbLogger()
        trainer = pl.Trainer(max_epochs=200, logger=wandb_logger, gpus=1)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
