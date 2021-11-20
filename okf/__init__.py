'''Optimized Kalman Filter: optimize your KF parameters from data of system-states and observations.'''
from okf.model import OKF
from okf.optimizer import train_models, train, split_train_valid, test_model, analyze_test_results, display_tracking
from okf import utils
from okf import example