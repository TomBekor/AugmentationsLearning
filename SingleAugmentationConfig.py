import numpy as np
import torch.nn as nn
import torch.optim as optim
import kornia as K
from kAugmentations.kContrast import *
from kAugmentations.kTranslateX import *
from kAugmentations.kRotation import *
from kAugmentations.kBrightness import *




class SingleAugmentationConfig:
    def __init__(self,
                 augmentation_name,
                 main_parameter_name,
                 aug_bounds,
                 param_linspace,
                 target_param_val,
                 target_aug_constructor,
                 target_aug_constructor_args,
                 kAugmentation,
                 init_param_val,
                 optimizer_constructor,
                 optimizer_constructor_args,
                 scheduler_constructor,
                 scheduler_constructor_args,
                 scheduler_warmup,
                 scheduler_freq,
                 use_scheduler,
                 early_stopping):
        self.augmentation_name = augmentation_name
        self.main_parameter_name = main_parameter_name
        self.aug_bounds = aug_bounds
        self.param_linspace = param_linspace
        self.target_param_val = target_param_val
        self.target_aug_constructor = target_aug_constructor
        self.target_aug_constructor_args = target_aug_constructor_args
        # self.target_aug_constructor_args[self.main_parameter_name] = target_param_val
        self.kAugmentation = kAugmentation
        self.init_param_val = init_param_val
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_constructor_args = optimizer_constructor_args
        self.scheduler_constructor = scheduler_constructor
        self.scheduler_constructor_args = scheduler_constructor_args
        self.scheduler_warmup = scheduler_warmup
        self.scheduler_freq = scheduler_freq
        self.use_scheduler = use_scheduler
        self.early_stopping = early_stopping


contrast_config = SingleAugmentationConfig(
    augmentation_name = 'contrast',
    main_parameter_name = 'contrast_factor',
    aug_bounds = (0.0, None),
    param_linspace = np.linspace(0.0,10.,400),
    target_param_val = 2.,
    target_aug_constructor = K.enhance.AdjustContrast,
    target_aug_constructor_args = {'contrast_factor': 2.},
    kAugmentation = kContrast,
    init_param_val = 1.,
    optimizer_constructor = optim.SGD,
    optimizer_constructor_args = {
        'lr': .1,
        'momentum': 0.9,
    },
    scheduler_constructor = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_constructor_args = {
        'T_max':100,
        'eta_min':0.1,
        'verbose':True,
    },
    scheduler_warmup = 100,
    scheduler_freq = 10,
    use_scheduler=False,
    early_stopping = 250,
)

translateX_config = SingleAugmentationConfig(
    augmentation_name = 'translateX',
    main_parameter_name = 'translation',
    aug_bounds = (None, None),
    param_linspace = np.linspace(-100,100,400),
    target_param_val = 10.0,
    target_aug_constructor = K.geometry.transform.Translate,
    target_aug_constructor_args = {'translation': torch.Tensor([[10.0, 0]]),},
    kAugmentation = kTranslateX,
    init_param_val = 0.0,
    optimizer_constructor = optim.SGD,
    optimizer_constructor_args = {
        'lr': 80,
        'momentum': 0.5,
    },
    scheduler_constructor = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_constructor_args = {
        'T_max':100,
        'eta_min':10,
        'verbose':True,
    },
    scheduler_warmup = 0,
    scheduler_freq = 10,
    use_scheduler=True,
    early_stopping = 250,
)


rotation_config = SingleAugmentationConfig(
    augmentation_name = 'rotation',
    main_parameter_name = 'angle',
    aug_bounds = (-360., 360.),
    param_linspace = np.linspace(0., 360.,200),
    target_param_val = 30.,
    target_aug_constructor = K.geometry.transform.Rotate,
    target_aug_constructor_args = {'angle': torch.Tensor([30.])},
    kAugmentation = kRotation,
    init_param_val = 0.,
    optimizer_constructor = optim.SGD,
    optimizer_constructor_args = {
        'lr': 100,
        'momentum': 0.99,
    },
    scheduler_constructor = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_constructor_args = {
        'T_max':100,
        'eta_min':10,
        'verbose':True,
    },
    scheduler_warmup = 100,
    scheduler_freq = 10,
    use_scheduler=True,
    early_stopping = 500,
)


brightness_config = SingleAugmentationConfig(
    augmentation_name = 'brightness',
    main_parameter_name = 'brightness_factor',
    aug_bounds = (0.0, 1.0),
    param_linspace = np.linspace(0., 1.,200),
    target_param_val = .25,
    target_aug_constructor = K.enhance.AdjustBrightness,
    target_aug_constructor_args = {'brightness_factor': .25},
    kAugmentation = kBrightness,
    init_param_val = 0.0,
    optimizer_constructor = optim.SGD,
    optimizer_constructor_args = {
        'lr': .01,
        'momentum': 0.9,
    },
    use_scheduler=True,
    scheduler_constructor = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_constructor_args = {
        'T_max':100,
        'eta_min':0.001,
        'verbose':True,
    },
    scheduler_warmup = 100,
    scheduler_freq = 10,
    early_stopping = 500,
)