import torch
import numpy as np

import matplotlib.pyplot as plt

def plot_helper(ax, a, b, timestamps, e, p, label):
    ax[a, b].scatter(timestamps, e, label=label)
    ax[a, b].fill_between(
        timestamps.reshape(
            -1,
        ),
        (3 * p).reshape(
            -1,
        ),
        (-3 * p).reshape(
            -1,
        ),
        color="g",
        alpha=0.2,
    )
    ax[a, b].legend(loc="upper right")

def three_element_3_sigma_plotter(ax_overall, row_id, label, errs, sigmas, ts): 
    plot_helper(ax_overall, row_id, 0, ts, errs[:, 0], sigmas[:, 0], label = label + "_x_err")
    plot_helper(ax_overall, row_id, 1, ts, errs[:, 1], sigmas[:, 1], label = label + "_y_err")
    plot_helper(ax_overall, row_id, 2, ts, errs[:, 2], sigmas[:, 2], label = label + "_z_err")

class ThreeDimensionalCartesianPlotter:
    def __init__(self, fig_title):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d', aspect='equal')
        self.ax.view_init(azim=-135 + 180)
        self.fig.suptitle(fig_title)

        self.fig_xy = plt.figure()
        self.ax_xy = self.fig_xy.add_subplot()
        self.fig_xy.suptitle(fig_title + "_xy")
        plt.grid(visible=True)

        self.fig_xz = plt.figure()
        self.ax_xz = self.fig_xz.add_subplot()
        self.fig_xz.suptitle(fig_title + "_xz")
        plt.grid(visible=True)

        self.fig_yz = plt.figure()
        self.ax_yz = self.fig_yz.add_subplot()
        self.fig_yz.suptitle(fig_title + "_yz")
        plt.grid(visible=True)

    def add_scatter(self, cart_coords : np.ndarray, label : str, color : str):
        """ plotting helper for 3-dimensional cartesian coordinates """
        self.ax.scatter(cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2], color=color, label=label)
        self.ax.set_xlabel('x (m)', fontsize=14)
        self.ax.set_ylabel('y (m)', fontsize=14)
        self.ax.set_zlabel('z (m)', fontsize=14)
        self.ax.legend(loc="upper right", fontsize=18, markerscale=2)

        self.ax_xy.scatter(cart_coords[:, 0], cart_coords[:, 1], color=color, label=label)
        self.ax_xy.set_xlabel('x (m)')
        self.ax_xy.set_ylabel('y (m)')
        self.ax_xy.legend(loc="upper right")
        self.ax_xy.text(cart_coords[:, 0][0], cart_coords[:, 1][0], "start")
        self.ax_xy.text(cart_coords[:, 0][-1], cart_coords[:, 1][-1], "end")

        self.ax_xz.scatter(cart_coords[:, 0], cart_coords[:, 2], color=color, label=label)
        self.ax_xz.set_xlabel('x (m)')
        self.ax_xz.set_ylabel('z (m)')
        self.ax_xz.legend(loc="upper right")
        self.ax_xz.text(cart_coords[:, 0][0], cart_coords[:, 2][0], "start")
        self.ax_xz.text(cart_coords[:, 0][-1], cart_coords[:, 2][-1], "end")

        self.ax_yz.scatter(cart_coords[:, 1], cart_coords[:, 2], color=color, label=label)
        self.ax_yz.set_xlabel('y (m)')
        self.ax_yz.set_ylabel('z (m)')
        self.ax_yz.legend(loc="upper right")
        self.ax_yz.text(cart_coords[:, 1][0], cart_coords[:, 2][0], "start")
        self.ax_yz.text(cart_coords[:, 1][-1], cart_coords[:, 2][-1], "end")

class BoxPlotter:
    def __init__(self, fig_title):
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(fig_title)

        # declare empty list of 1D box-plottable vectors to compare
        self.data_vec = []

        # declare empty list of corresponding labels
        self.label_vec = []

    def add_data_vec(self, data : np.ndarray, label : str):
        self.data_vec.append(data)
        self.label_vec.append(label)

    def plot(self):
        self.ax.boxplot(self.data_vec, showfliers=False)
        self.ax.set_xticklabels(self.label_vec)