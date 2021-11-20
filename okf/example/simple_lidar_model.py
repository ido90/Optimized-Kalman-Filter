'''
Definition of the simple 2D lidar problem, used to create the KF model.

The system state is its location and velocity (x,y,vx,vy).
The dynamics model is a constant-velocity motion.
The observation model is a single sensor with known location that observes the location (x,y).
The goal is to predict either the state (x,y,vx,vy) or the location (x,y).

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
    # x,y -> x,y,vx,vy
    return torch.cat((z, torch.zeros(2,dtype=torch.double)))

def loss_fun(location_only=True):
    # MSE over either the location or the whole state (location+velocity)
    if location_only:
        return lambda pred, x: ((pred[:2]-x[:2])**2).sum()
    return lambda pred, x: ((pred-x)**2).sum()

def model_args():
    return dict(
        dim_x = 4,
        dim_z = 2,
        init_z2x = initial_observation_to_state,
        F = get_F(),
        H = get_H(),
        loss_fun = loss_fun(),
    )
