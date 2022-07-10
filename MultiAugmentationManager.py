from select import select
from turtle import st
from typing_extensions import Self
from SingleAugmentationConfig import SingleAugmentationConfig
from typing import List
import kornia as K

class MultiAugmentationManager:
    def __init__(self, sa_configs: List[SingleAugmentationConfig]):
        self.sa_configs = sa_configs
        self.create_run_name()
        self.create_grid_title()
        self.augs_names = []
        for sa_conf in self.sa_configs:
            self.augs_names.append(sa_conf.augmentation_name)


    def create_run_name(self):
        aug_names = ''
        inits = '('
        targets = '('
        for sa_conf in self.sa_configs:
            aug_names += sa_conf.augmentation_name + '-'
            inits += str(sa_conf.init_param_val) + ','
            targets += str(sa_conf.target_param_val) + ','
        aug_names = aug_names[:-1]
        inits = inits[:-1] + ')'
        targets = targets[:-1] + ')'
        self.run_name = f'{aug_names}_inits-{inits}_targets-{targets}'

    def create_grid_title(self):
        self.grid_title = ''
        for sa_conf in self.sa_configs:
            self.grid_title += sa_conf.augmentation_name + ' + '
        self.grid_title = self.grid_title[:-3]

    def get_target_augmentations(self):
        tar_aug_list = []
        for sa_conf in self.sa_configs:
            tar_aug_list.append(sa_conf.target_aug_constructor(**sa_conf.target_aug_constructor_args))
        return K.augmentation.container.ImageSequential(*tar_aug_list)

    def get_augmentations_model(self):
        k_aug_list = []
        for sa_conf in self.sa_configs:
            k_aug_list.append(sa_conf.kAugmentation(init_param=sa_conf.init_param_val))
        return K.augmentation.ImageSequential(*k_aug_list)

    def init_optimizers(self, model):
        self.optimizers = {}
        for sa_conf, child in zip(self.sa_configs, model): # important to iterate at the same order of the model initialization!
            self.optimizers[sa_conf.augmentation_name] = sa_conf.optimizer_constructor(child.parameters(), **sa_conf.optimizer_constructor_args)

    def init_schedulers(self):
        self.schedulers = {}
        for sa_conf in self.sa_configs:
            self.schedulers[sa_conf.augmentation_name] = sa_conf.scheduler_constructor(
                self.optimizers[sa_conf.augmentation_name], **sa_conf.scheduler_constructor_args
                )

    def get_current_params_str(self, model):
        s = ''
        for sa_conf, child in zip(self.sa_configs, model):
            s += f'{sa_conf.main_parameter_name}: {child.get_param_val():.3f}, '
        s = s[:-2]
        return s

    def optimizers_zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def optimizers_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def schedulers_step(self, i):
        for sa_conf in self.sa_configs:
            if sa_conf.use_scheduler:
                if i % sa_conf.scheduler_freq == sa_conf.scheduler_freq - 1 and i > sa_conf.scheduler_warmup:
                    self.schedulers[sa_conf.augmentation_name].step()

    def init_params_progress(self):
        self.params_progress = {}
        for aug_name in self.augs_names:
            self.params_progress[aug_name] = []

    def get_current_param_vals(self, model):
        param_vals = {}
        for aug_name, child in zip(self.augs_names, model):
            param_vals[aug_name] = child.get_param_val()
        return param_vals

    def update_params_progrogress(self, model):
        current_params_dict = self.get_current_param_vals(model)
        for aug_name, param_val in current_params_dict.items():
            self.params_progress[aug_name].append(param_val)

    def init_lrs_progress(self):
        self.lrs_progress = {}
        for aug_name in self.augs_names:
            self.lrs_progress[aug_name] = []

    def get_current_lrs(self):
        lrs = {}
        for aug_name, scheduler in self.schedulers.items():
            lrs[aug_name] = scheduler.get_last_lr()
        return lrs

    def update_lrs_progrogress(self):
        current_lrs_dict = self.get_current_lrs()
        for aug_name, lr in current_lrs_dict.items():
            self.lrs_progress[aug_name].append(lr)