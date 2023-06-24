# score_po
This repository contains code for "Fighting Uncertainty with Gradients: Offline Reinforcement Learning via Diffusion Score Matching".

---

### Uncertainty-Penalized Optimal Control

We perform uncertainty penalization optimal control problems of the folloinwg form,

![image](https://github.com/hjsuh94/score_po/assets/22463195/a7348511-8db1-4b55-a0f5-696fd0224703)

where θ are policy parameters, r is the reward, and p is the perturbed empirical distribution of data that encourages rollout trajectories to stay close to data. 

### Score-Guided Planning (SGP)

Our codebase supports general feedback policy optimization, but examples mainly evolve around open-loop planning. This is a special case of uncertainty-penalized optimal control where the policy is parametrized as an open-loop sequence of inputs.

---

### How to Run 

#### Installation 
This repo is mainly written in `torch`, and heavily uses `wandb` and `hydra-core`. For some robotic examples, [`drake`](https://drake.mit.edu/) might be required as a dependency. 

To install these dependencies and set the path, simply run
```
python -m pip install -r requirements.txt
```
after cloning the repo, and add the python path to `~/.bashrc`. Since this repo relies on calling lines such as `import examples`, we recommend putting this line at the
end of the bashrc file. 
```
export PYTHONPATH=${HOME}/score_po:${PYTHONPATH}
```

#### Running with Hydra 
We use hydra for our examples, and users are required to have a config file. Add your own user config file under `config/user` for each example, and modify the config files to have your user name.

For example, to run `examples/cartpole/learn_model.py`, 
1. Add a profile to `examples/cartpole/config/user` as `new_user.yaml`, following patterns of `terry.yaml`.
2. In `examples/cartpole/config/learn_model.yaml`, set
```
defaults:
  - user: new_user
```
3. Run `python examples/cartpole/learn_model.py` from the cloned directory. 

#### Running Tests

We use `pytest` for testing and CI. To run tests, do
```
pytest .
```
from the cloned directory.
 
---

### Examples 

All the examples can be found in the examples folder with instructions on how to run.

- Simple1D 
- Cart-pole system 
- The pixel-space single integrator: use branch `pixels_glen`. 
- D4RL Mujoco Benchmark
- Box-Keypoint Pushing Example: for hardware code, use branch `lcm_hardware`. 

---

### Citations 

