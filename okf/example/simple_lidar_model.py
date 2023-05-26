'''
The KF model (OKF object) can be adjusted to any new filtering problem via its arguments.
This module provides an example for such arguments, corresponding to a simplified 2D lidar problem.

Note:
    - model_args() returns the arguments for the constructor OKF().
    - F is of size (dim_x, dim_x) = (4, 4).
    - H is of size (dim_z, dim_x) = (2, 4).
    - Trajectory initialization:
        Either init_z2x or x0 need to be defined.
        If x0 is defined, then every trajectory is initialized to the initial guess x=x0.
        Otherwise, the state is initialized to x=init_z2x(z) - only after an observation is received.
        In this example, we prefer the latter: we derive the initialization from the first observation.

Specifically, for the simplified 2D lidar problem:
    - The system state is its location and velocity (x,y,vx,vy).
    - The dynamics model is a constant-velocity motion.
    - The observation model is a single sensor with known location that observes the location (x,y).
    - The goal is to predict either the state (x,y,vx,vy) or the location (x,y).

Written by Ido Greenberg, 2021
'''

import numpy as np
import torch

def get_F():
    # x,y,vx,vy
    return torch.tensor([
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=torch.double)

def get_H():
    # x,y,vx,vy -> x,y
    return torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
    ], dtype=torch.double)

def initial_observation_to_state(z):
    # x,y -> (x=x, y=y, vx=0, vy=0)
    return torch.cat((z, torch.zeros(2,dtype=torch.double)))

def loss_fun(location_only=True):
    # MSE over either the location or the whole state (location & velocity)
    if location_only:
        return lambda pred, x: ((pred[:2]-x[:2])**2).sum()
    return lambda pred, x: ((pred-x)**2).sum()

def model_args():
    return dict(
        dim_x = 4,
        dim_z = 2,
        init_z2x = initial_observation_to_state,  # this could be replaced by explicit initialization - x0=...
        F = get_F(),
        H = get_H(),
        loss_fun = loss_fun(),
    )
