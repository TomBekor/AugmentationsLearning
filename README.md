# Experiment #1

## rotation-init-0.0_target-45.0
![init-0.0_target-45.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-0.0_target-45.0.gif)
## rotation-init-50.0_target-90.0
![init-50.0_target-90.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-50.0_target-90.0.gif)
## rotation-init-0.0_target-100.0
![init-0.0_target-100.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-0.0_target-100.0.gif)
![loss_map](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/init-0.0_target-100.0/results/loss_map.png)
![training_loss](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/init-0.0_target-100.0/results/training_loss.png)
![parameter_progress](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/init-0.0_target-100.0/results/parameter_progress.png)
![lr_change](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/init-0.0_target-100.0/results/learning_rate.png)
## brightness-init-0.0_target-0.5
![brightness-init-0.0_target-100.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/brightness-init-0.0_target-0.5.gif)
![loss_map](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/brightness-init-0.0_target-0.5/results/loss_map.png)
![training_loss](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/brightness-init-0.0_target-0.5/results/training_loss.png)
![parameter_progress](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/brightness-init-0.0_target-0.5/results/parameter_progress__brightness_factor.png)
![lr_change](https://github.com/TomBekor/AugmentationsLearning/blob/master/figures/brightness-init-0.0_target-0.5/results/learning_rate.png)





## Explenations:
(Past)<br>
Trying to learn the parameter of the rotation augmentation on rotated Tiny ImageNet. <br>
Some experiments work (first 2), and some still have convergence problems (last 2). <br>
The padding plays a strong role in the L2 Loss function...<br>
On the last example, the learning rate excessively increased so <br>
the optimizer will jump over very high hills in the loss. <br>

Cosine scheduler with warm start added.<br>
init-0.0_target-100.0 - managed to converge, but very sensitive to hyper-parameters.<br>
Figures added: loss map, training loss, learning rate change.<br>
Next step: Improve robustness for hyper-parameters / generalize to other augmentations.<br> <br>

(Present)<br>
Hyper-parameter robustness improved, also robustness for the augmentation parameter initialization.<br>
Experiment also working with the Brightness augmentation. <br>
Figures added: 1. augmentation parameter history, 2. on the loss map, orange line which indicates the places where the model visited while training.
Generalized the pipeline to work with all sort of augmentations, just need to change the config cell (currently not so trivial) (still need one last change to create the loss map).
