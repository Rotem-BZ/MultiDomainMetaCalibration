# Meta-Calibration: Learning of Model Calibration Using Differentiable Expected Calibration Error

Calibration of neural networks is a topical problem that is becoming more and more important as neural networks increasingly underpin real-world applications. The problem is especially noticeable when using modern neural networks, for which there is a significant difference between the confidence of the model and the probability of correct prediction. Various strategies have been proposed to improve calibration, yet accurate calibration remains challenging. We propose a novel framework with two contributions: introducing a differentiable surrogate for expected calibration error (DECE) that allows calibration quality to be directly optimised, and a meta-learning framework that uses DECE to optimise for validation set calibration with respect to model  hyper-parameters. The results show that we achieve competitive performance with state-of-the-art calibration approaches. Our framework opens up a new avenue and toolset for tackling calibration, which we believe will inspire further work in this important challenge.

Our implementation extends the [implementation](https://github.com/torrvision/focal_calibration) for paper [*Calibrating Deep Neural Networks using Focal Loss*](https://arxiv.org/abs/2002.09437) from Mukhoti et al. You can find further useful information there.


<p align="center"><img src='DECEandECEcorrelations.png' width=700></p>

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
The approach is implemented in PyTorch and its dependacies are listed in [environment.yml](environment.yml).

### Datasets
CIFAR-10 and CIFAR-100 datasets will be downloaded automatically.

## Experiments

You can train and evaluate a model with meta-calibration using the following commands:
```
python train.py --dataset cifar10 --model resnet18 --loss cross_entropy --save-path Models/ --exp_name rn18_c10_meta_calibration --meta_calibration

python evaluate.py --dataset cifar10 --model resnet18 --save-path Models/ --saved_model_name rn18_c10_meta_calibration_best.model --exp_name rn18_c10_meta_calibration
```

## Multidomain training
The following libraries need to be installed first to be able to generate the corruptions: `skimage`, `wand` and `cv2`. CIFAR-C can be downloaded from https://github.com/hendrycks/robustness.

To run multidomain training, it enough to use argument `--multi_domain` and evaluation is done using `evaluate_domains.py` similar to `evaluate.py`. 