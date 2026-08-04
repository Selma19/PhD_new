"""Draws the figures for the paper.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from database import Kernel_db
from database.kernel.fill_db import read_kernel

def plot_kernel(ax, k_out, k_type, k_meth):
    """Plots the kernel function on `ax`."""
    k = read_kernel(k_out, k_type, k_meth)
    ax.plot(np.abs(k), '.')

"""Figure 1:

?
"""
