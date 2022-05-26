# Experiment #1
Trying to learn the parameter of the rotation augmentation on rotated Tiny ImageNet. <br>
Some experiments work (first 2), and some still have convergence problems (last 2). <br>
The padding plays a strong role in the L2 Loss function...<br>
On the last example, the learning rate excessively increased so <br>
the optimizer will jump over very high hills in the loss. <br>
Next step: add scheduler.
## init-0.0_target-45.0
![init-0.0_target-45.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-0.0_target-45.0.gif)
## init-50.0_target-90.0
![init-50.0_target-90.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-50.0_target-90.0.gif)
## init-300.0_target-45.0
![init-300.0_target-45.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-300.0_target-45.0.gif)
## init-0.0_target-100.0
![init-0.0_target-100.0](https://github.com/TomBekor/AugmentationsLearning/blob/master/gifs/init-0.0_target-100.0.gif)
