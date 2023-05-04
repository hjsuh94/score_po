# Overview
This folder contains the cartpole example. We do the following things for the robot:
1. Generate the dataset consisting tuples (x, u, x_next).
2. Train the dynamics models
   - If we use single shooting method, then we train a dynamics model `x_next = f̅(x, u)` through regression, and a score function `log p(x, u)`
   - If we use direct collocation method, then we train a score function `log p(x, u, x_next)`
3. Run the trajectory optimization (either single shooting or direct collocation) to swing up the cart pole.

## Single shooting
If you decide to run the single shooting method, then run the code this way:
1. Run `learn_model.py`. Make sure your `train_model` is set to `True` (either in the learn_model.yaml or through the command line). This will generate the training dataset with (x, u, x_next), and learn the dynamics model `x_next = ̅f(x, u)` through regression.
2. Run `score_training.py`. Make sure `train.xu` is set to `True` (either in score_training.yaml or through the command line). Also make sure that `dataset.load_path` is set to the path of the dataset generated in step 1. This will learn a score function for `log p(x, u)`.
3. Run `swingup.py`. Make sure `single_shooting` is set to `True` (either in swingup.yaml or through the command line). Also you need to set `dynamics_load_ckpt` to the model in step 1, and `score_estimator_xu_load_ckpt` to the one in step 2. This will generate the trajectory to swing up a cart pole. I would recommend using a sweep of `beta` parameter. You will see that with `beta=0`, the trajectory will likely go outside of the box where we generate the data; if `beta` is set to a very large value then we cannot swing up the cart pole as the `β log p(x, u)` term will dominate the cost.

## Direct collocation
If you decide to run the direct collocation method, then run the code this way:
1. Run `learn_model.py`. You can set `train_model` is set to `False` (as you don't need the regression model `x_next = ̅f(x, u)`).
2. Run `score_training.py`. Make sure `train.xu` is set to `False` (either in score_training.yaml or through the command line). Also make sure that `dataset.load_path` is set to the path of the dataset generated in step 1. This will learn a score function for `log p(x, u, x_next)`.
3. Run `swingup.py`. Make sure `single_shooting` is set to `False` (either in swingup.yaml or through the command line). Also you need to set `score_estimator_xux_load_ckpt` to the model in step 2. 