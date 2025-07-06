# Soft Actor-Critic with Backstepping-Pretrained DeepONet for control of PDEs
The code for the paper *Soft Actor-Critic with Backstepping-Pretrained
DeepONet for control of PDEs*. 

## Preliminary
Our demonstration code is based on Python 3.11 and utilizes several key third-party libraries, including 
* PyTorch 2.7.1
* SciPy 1.14.1
* pandas 2.2.3 
* DeepXDE 1.13.2
* Gym version 0.29.1
* Numpy version 2.2.4
* Stable Baselines3 version 2.2.1


# Code Description
* "Hyperbolic_DeepONet" and "reactionDiffusion_DeepONet": Used for dataset generation and training the DeepONet model.  
* "Hyperbolic_training" and "reactionDiffusion_training": Used for reinforcement learning training, including SAC (Soft Actor-Critic), NOSAC , and NOSAC_training.
* "Hyperbolic_test" and "reactionDiffusion_test": Used to store all images generated during the program execution.
* "pde_control_gym": It includes rewards and environments.
