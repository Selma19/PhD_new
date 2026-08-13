"""Draws the figures for the paper.
"""

import os, json
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from database import Kernel_db

k_type = 'param1'
k_meth = 'nested_sampling'
filtering_method = 'unfiltered'

db = Kernel_db()
db.connect()
print('yo')
exit()
# get the rows id (main_key) in the Main table
# that meet our needs
rows = db.cur.execute("""
    SELECT id, agent, coh
    FROM Main
    WHERE
        filtering_method = ?
    GROUP_BY
        coh
""", filtering_method).fetchall()
for row in rows:
    print(row)
    print()
exit()

# for each row, get the kernel output, then
# the kernel parameters and their uncertainties
uncertainties = {}
means = {}
for row in rows:
    k_out = db.cur.execute("""
        SELECT kernel_output
        FROM Kernels
        WHERE
            main_key = ?
        AND
            kernel_type = ?
        AND
            kernel_method = ?
    """, (row[0], k_type, k_meth))

    points, log_w = json.loads(k_out)
    prob_w = softmax(log_w)

    for name, val in points.items():
        tab = np.array(val)
        mean = np.sum(prob_w * tab)
        std = np.sum(prob_w * tab ** 2) - mean ** 2

        if name in uncertainties:
            uncertainties[name].append(std)
            means[name].append(mean)
        else:
            uncertainties[name] = std
            means[name] = mean
        
# plot the figure 1:
# one panel per parameter;
# coherence as x axis
# parameter value as y axis, organized as violins or candles
params = list(means)

fontsize = 14
for param in params:
    fig, ax = plt.subplots(1, 1)
    ax.set_title(param, fontsize=fontsize)

    # dic[coh] = [mean of each agent]
    dic = {}

    for row, mean in zip(rows, means[name]):
        coh = row[2]
        if coh in dic:
            dic[coh].append(mean)
        else:
            dic[coh] = [mean]

    # plot
    n_cohs = len(dic)

