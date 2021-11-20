# Optimized Kalman Filter

Get an optimized Kalman Filter from data of system-states and observations.

This package implements the algorithm introduced in the paper [Kalman Filter Is All You Need: Optimization Works When Noise Estimation Fails](https://arxiv.org/abs/2104.02372), by Greenberg, Mannor and Yannay.

- [Background](#background--the-kalman-filter)
- [When to use](#when-to-use-this-package)
- [Why to use](#why-to-use)
- [How to use](#how-to-use)

## Background: the Kalman Filter

The Kalman Filter (KF) is a popular algorithm for filtering problems such as state estimation, smoothing, tracking and navigation. For example, consider tracking a plane using noisy measurements (observations) from a radar. Every time-step, we try to predict the motion of the plane, then receive a new measurement from the radar and update our belief accordingly.

| <img src="https://idogreenberg.neocities.org/linked_images/KF_illustration.png" width="360"> |
| :--: |
| An illustration of a single step of the Kalman Filter: predict the next state (black arrow); receive a new observation (green ellipse); update your belief about the state (mixing the two right ellipses) |

| <img src="https://idogreenberg.neocities.org/linked_images/KF_diagram.png" width="480"> |
| :--: |
| A diagram of the Kalman Filter algorithm |

To tune the KF, one has to determine the parameters representing the measurement (observation) noise and the motion-prediction noise, expressed as covariance matrices R and Q. Given a dataset of measurements {z_t} (e.g. from the radar), tuning these parameters may be a difficult task, and has been studied for many decades. However, given data of *both* measurements {z_t} and true-states {x_t} (e.g. true plane locations), the parameters R,Q are usually estimated from the data as the sample covariance matrices of the noise.

## When to use this package

When you want to build a Kalman Filter for your problem, and you have a training dataset with sequences of both states {x_t} and observations {z_t}.

## Why to use

Tuning the KF parameters through noise estimation (as explained above) yields optimal model predictions - under the KF assumptions. However, as shown in the paper, whenever the assumptions do not hold, optimization of the parameters may lead to more accurate predictions. Since most practical problems do not satisfy the assumptions, and since assumptions violations are often not noticed at all by the user, optimization of the parameters is a good practice whenever a corresponding dataset is available.

## How to use

TODO
