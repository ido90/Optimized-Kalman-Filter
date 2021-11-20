'''
A simulator of data for the simple 2D lidar problem, used as an example to use for KF optimization.

The system state is its location and velocity (x,y,vx,vy).
The targets trajectories consist of alternating straight intervals and turns, with random accelerations.
The observations are noisy samples of the target location (x,y), as measured by a single sensor with known location.

Written by Ido Greenberg, 2021
'''

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .. import utils


############   DATA API   ############

def load_data(fpath='data/simple_lidar_data.pkl'):
    with open(fpath, 'rb') as fd:
        X, Z = pkl.load(fd)
    return X, Z

def get_trainable_data(X, Z):
    '''Convert the list of data-frames into numpy arrays.'''
    return [x.values.astype(np.double) for x in X], [z.values.astype(np.double) for z in Z]

def display_data(X, Z):
    axs = utils.Axes(8, 4, (5, 3.5))

    utils.plot_quantiles([len(x) for x in X], axs[0], showmeans=True)
    axs.labs(0, 'Quantile [%]', 'Number of time steps', f'Lengths distribution over {len(X):d} trajectories')

    for i in range(6):
        axs[1].plot(X[i]['x'], X[i]['y'], '-', label=str(i))
    axs[1].set_title('A sample of trajectories', fontsize=14)
    axs[1].legend()

    for i in range(6):
        ax = axs[2+i]
        ax.plot(X[i]['x'], X[i]['y'], 'k.-', label='state')
        ax.plot(Z[i]['zx'], Z[i]['zy'], 'r.', label='observation')
        axs.labs(2+i, 'x', 'y', f'Target {i}')
        ax.legend()

    plt.tight_layout()


############   SIMULATOR   ############

def rand_range(a,b):
    return (b - a) * np.random.rand() + a

def rand_range_sym(d):
    return d * (2*np.random.rand()-1)

def simulate_data(n_targets=1000, x0=600, y0=600, v0=(30,90),
                  n_intervals=(4,7), int_len=(5,9), ar_sigma=2, at_sigma=15,
                  noise_r=2, noise_t=1.5*np.pi/180, fpath='data/simple_lidar_data.pkl'):
    X, Z = [], []
    for i in range(n_targets):
        x, y, vx, vy, zx, zy = [],[],[],[],[],[]
        # init state
        x.append(rand_range_sym(x0))
        y.append(rand_range_sym(y0))
        vtheta = rand_range(0,2*np.pi)
        v = rand_range(*v0)
        vx.append(v*np.cos(vtheta))
        vy.append(v*np.sin(vtheta))
        r = np.linalg.norm((x[-1],y[-1])) + np.random.normal(0, noise_r)
        theta = np.arctan2(y[-1], x[-1]) + np.random.normal(0, noise_t)
        zx.append(r*np.cos(theta))
        zy.append(r*np.sin(theta))
        # intervals
        n_ints = np.random.randint(*n_intervals)
        for interval in range(n_ints):
            interval_len = np.random.randint(*int_len)
            ar = np.random.normal(0, ar_sigma)
            at = rand_range_sym(at_sigma)
            for t in range(interval_len):
                # simulate motion
                v = np.linalg.norm((vx[-1],vy[-1]))
                ax = ar*vx[-1]/v - at*vy[-1]/v
                ay = ar*vy[-1]/v + at*vx[-1]/v
                x.append(x[-1]+vx[-1]+0.5*ax)
                y.append(y[-1]+vy[-1]+0.5*ay)
                vx.append(vx[-1]+ax)
                vy.append(vy[-1]+ay)
                # simulate observation
                r = np.linalg.norm((x[-1],y[-1])) + np.random.normal(0, noise_r)
                theta = np.arctan2(y[-1], x[-1]) + np.random.normal(0, noise_t)
                zx.append(r*np.cos(theta))
                zy.append(r*np.sin(theta))
        # save target
        XX = pd.DataFrame(dict(
            x = x,
            y = y,
            vx = vx,
            vy = vy,
        ))
        ZZ = pd.DataFrame(dict(
            zx = zx,
            zy = zy,
        ))
        X.append(XX)
        Z.append(ZZ)
    if fpath:
        with open(fpath, 'wb') as fd:
            pkl.dump((X,Z), fd)
    return X, Z
