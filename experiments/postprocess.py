# -*- coding: utf-8 -*- 
# @Time : 2023/6/10 13:51 
# @Author : lepold
# @File : postprocess.py

import sys

sys.path.append("/public/home/ssct004t/project/zenglb/DetailedDTB/")

import matplotlib.pyplot as plt
from matplotlib import gridspec, rc_file, rc, rcParams
import re
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import h5py

rc_file(r'../utils/plotstyle.rc')
# rcParams['font.sans-serif'] = ['Arial']
# rc('text', usetex=True)

sample_n = 900
document_time = 10
bold_time = 300

def typical_raster(res_path: str, region_idx=None, fig=None):
    file_nmae = re.compile(r"spike_.+_assim_\d+.npy")
    blocks = [name for name in os.listdir(res_path) if file_nmae.fullmatch(name)]
    blocks = sorted(blocks)
    spike_path = os.path.join(res_path, blocks[-1])
    print(f"\nload spike from {spike_path}")
    Spike = np.load(spike_path)
    ax = {}
    ax[0] = fig.add_axes([0.08, 0.55, 0.2, 0.38], frameon=True)
    ax[1] = fig.add_axes([0.32, 0.55, 0.2, 0.38], frameon=True)
    Spike = Spike[-2:, :, :].reshape((2 * 800, -1))  # steps: 800
    if region_idx is None:
        spike_index = np.concatenate([np.arange(i * sample_n, (i + 1) * sample_n) for i in [43, 103]])
    else:
        assert len(region_idx) == 2
        spike_index = np.concatenate([np.arange(i * sample_n, (i + 1) * sample_n) for i in region_idx])
    spike_events = [Spike[:, i].nonzero()[0] for i in spike_index]
    colors = ["tab:blue", "tab:red"]
    total = sample_n
    sample_dis = np.array([80, 20, 80, 20, 20, 10, 60, 10], dtype=np.int32) * 3  # sum=300
    names = ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    for j in range(len(region_idx)):
        s = 0
        for i, size in enumerate(sample_dis):
            e = s + size
            color = colors[i % 2]
            y_inter = (s + e) / 2 / total - 0.02
            fr = Spike[:, j*sample_n + s: j * sample_n + e].mean() * 1000
            ax[j].eventplot(spike_events[j*sample_n + s: j * sample_n + e], lineoffsets=np.arange(s, e), colors=color, linestyles="dashed")
            # xx, yy = Spike[:, s:e].nonzero()
            # yy = yy + s
            # ax.scatter(xx, yy, marker=',', s=1., color=color)
            s = e
            ax[j].text(0.6, y_inter, names[i] + f": {fr:.1f}Hz", color=color, fontsize=9, transform=ax[j].transAxes)
            ax[j].set_ylim([0, sample_n])
            ax[j].set_yticks([0, sample_n])
            ax[j].set_yticklabels([0, sample_n])
            ax[j].set_ylabel("Neuron")
            ax[j].set_xlim([0, 1600])
            ax[j].set_xticks([0, 1000, 1600])
            ax[j].set_xticklabels(["0.0", "1.0", "1.6"])
            ax[j].set_xlabel("Time (s)")
    ax[0].text(-0.1, 1.05, "A",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)


def typical_fr(res_path: str, voxel_idx=None, fig=None):
    file_nmae = re.compile(r"firing_.+_assim_\d+.npy")
    blocks = [name for name in os.listdir(res_path) if file_nmae.fullmatch(name)]
    blocks = sorted(blocks)
    firing_path = os.path.join(res_path, blocks[-1])
    print(f"\nload firing rate from {firing_path}")
    firing = np.load(firing_path)
    firing = firing[-5:, :, :].reshape((5 * 800, -1))
    assert voxel_idx is not None
    ax = {}
    gs = gridspec.GridSpec(2, 1)
    gs.update(left=0.08, right=0.52, top=0.52, bottom=0.4, hspace=0.1)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    ax[1] = fig.add_subplot(gs[1, 0], frameon=True)

    assert len(voxel_idx) == 2
    for i in range(2):
        fr = firing[:, voxel_idx[i]] * 1000
        rate_time_series_auto_kernel = gaussian_filter1d(fr, 5, axis=-1)
        ax[i].plot(rate_time_series_auto_kernel, color='0.2')
        ax[i].spines['right'].set_color('none')
        ax[i].spines['top'].set_color('none')
        if i == 0:
            ax[i].set_ylabel("Firing")
        if i == 1:
            ax[i].set_xticks([0, 4000])
            ax[i].set_xlabel([0, 4])
            ax[i].set_xlabel("Time (s)")

def heatmap_fc(res_path: str, fig=None):
    file_nmae = re.compile(r"bold_.+_assim.npy")
    # file_nmae = re.compile(r"firing_.+_assim_\d+.npy")  # compute fc using firing rate
    blocks = [name for name in os.listdir(res_path) if file_nmae.fullmatch(name)]
    assert len(blocks) == 1
    bold_sim = np.load(os.path.join(res_path, blocks[0]))
    bold_sim = bold_sim.reshape((-1, bold_sim.shape[-1]))
    file = h5py.File("../data/raw_data/NSR_data_May22.mat", "r")
    hpc_label = file["NSR_dti_HCPex_label"][:].squeeze()
    bold_exp = file["NSR_Resting_state"][:]
    if bold_exp.shape[0] > bold_exp.shape[1]:
        bold_exp = bold_exp.T
    print("bold shape", bold_sim.shape, bold_exp.shape)
    assert bold_sim.shape[1] == bold_exp.shape[1]
    N = bold_sim.shape[1]
    assert N == hpc_label.shape[0]
    uni_idx = np.unique(hpc_label).astype(np.int32)
    region_num = len(uni_idx)
    print(f"\ntotally {len(uni_idx)} brain region")
    bold_sim_region = np.zeros((bold_sim.shape[0], region_num), dtype=np.float32)
    bold_exp_region = np.zeros((bold_exp.shape[0], region_num), dtype=np.float32)
    invalid_index = []
    for idx in range(len(uni_idx)):
        index = np.where(hpc_label == uni_idx[idx])[0]
        if len(index)>0:
            bold_sim_region[:, idx] = bold_sim[:, index].mean(axis=-1)
            bold_exp_region[:, idx] = bold_exp[:, index].mean(axis=-1)
        else:
            invalid_index.append(idx)
    invalid_index = np.array(invalid_index, dtype=np.int32)
    fc_exp = np.corrcoef(bold_exp_region, rowvar=False)
    fc_exp[:, invalid_index] = np.nan
    fc_exp[invalid_index, :] = np.nan

    bold_sim_region = bold_sim_region[:bold_time]
    fc_sim = np.corrcoef(bold_sim_region, rowvar=False)
    fc_sim[:, invalid_index] = np.nan
    fc_sim[invalid_index, :] = np.nan

    fc_exp_flatten = fc_exp.flatten()
    fc_sim_flatten = fc_sim.flatten()
    p = np.corrcoef(fc_exp_flatten, fc_sim_flatten)[0, 1]

    ax = {}
    gs = gridspec.GridSpec(2, 1)
    gs.update(left=0.58, right=0.92, top=0.93, bottom=0.4, hspace=0.15)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    ax[1] = fig.add_subplot(gs[1, 0], frameon=True)
    ax[0].imshow(fc_exp, cmap='RdBu_r', interpolation='gaussian')
    ax[1].imshow(fc_sim, cmap='RdBu_r', interpolation='gaussian')
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    ax[0].set_title("FC (exp)")
    ax[1].set_title("FC (sim)")
    ax[0].text(-0.1, 1.05, "B",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    ax[1].text(-0.1, 1.05, "C",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[1].transAxes)
    ax[1].text(0.4, 1.2, f"p={p}",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[1].transAxes)


def statics(res_path, fig=None):
    ax = {}
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.1, right=0.92, top=0.35, bottom=0.08, wspace=0.2)
    ax[0] = fig.add_subplot(gs[0, 0], frameon=True)
    ax[1] = fig.add_subplot(gs[0, 1], frameon=True)
    ax[2] = fig.add_subplot(gs[0, 2], frameon=True)
    # ax[3] = fig.add_subplot(gs[0, 3], frameon=False)

    file_nmae = re.compile(r"firing_.+_assim_\d+.npy")
    blocks = [name for name in os.listdir(res_path) if file_nmae.fullmatch(name)]
    blocks = sorted(blocks)
    firing_path = os.path.join(res_path, blocks[-1])
    print(f"\nload firing rate from {firing_path}")
    firing = np.load(firing_path)
    firing = firing[-document_time:, :, :].reshape((document_time * 800, -1))

    file_nmae = re.compile(r"spike_.+_assim_\d+.npy")
    blocks = [name for name in os.listdir(res_path) if file_nmae.fullmatch(name)]
    blocks = sorted(blocks)
    spike_path = os.path.join(res_path, blocks[-1])
    print(f"\nload spike from {spike_path}")
    Spike = np.load(spike_path)
    Spike = Spike[-document_time:, :, :].reshape((document_time * 800, -1))  # steps: 800

    mean_fr = firing.mean(axis=0) * 1000
    ccs = firing.std(axis=0) / firing.mean(axis=0)
    cvs = []
    num_voxel = int(Spike.shape[-1] / sample_n)
    for ind in range(num_voxel):
        spike = Spike[:, ind * sample_n:(ind + 1) * sample_n]
        spike = spike.T.reshape(-1)
        pulse = np.nonzero(spike)[0]
        interval = pulse[1:] - pulse[:-1]
        cvs.append(interval.std() / interval.mean())
    cvs = np.array(cvs)

    cmap = plt.cm.viridis_r
    for idx, value in zip(range(3), [mean_fr, cvs, ccs]):
        cnts, values, bars = ax[idx].hist(value, bins=30, rwidth=0.8, log=True)
        newvalue = (value - value.min()) / (value.max() - value.min())
        for i, (cnt, value_data, bar) in enumerate(zip(cnts, values, bars)):
            bar.set_facecolor(cmap(newvalue[i]))
        # ax[idx].text(0.1, 1.0, f"min{value.min():.1f}->max{value.max():.1f}", transform=ax[idx].transAxes, va='top',
        #              fontsize=12,
        #              bbox={'color': 'r', 'alpha': 0.4, })
    ax[0].set_xlabel("Fr")
    ax[1].set_xlabel("CV")
    ax[2].set_xlabel("CC")
    ax[0].text(-0.1, 1.05, "D",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[0].transAxes)
    ax[1].text(-0.1, 1.05, "E",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[1].transAxes)
    ax[2].text(-0.1, 1.05, "F",
               fontdict={'fontsize': 11, 'weight': 'bold',
                         'horizontalalignment': 'left', 'verticalalignment':
                             'bottom'}, transform=ax[2].transAxes)


if __name__ == '__main__':
    # res_path = r"/public/home/ssct004t/project/zenglb/DetailedDTB/data/result_data/simulation_june12th"
    res_path = r"/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain/simuafterda_rest_thalamus_500m_202306120931/debug_202306121138_0.4_0.15_4_350_1e-8_1.21_30_0.02_0.05_0_0.08_0.5_0.1ms"
    fig = plt.figure(figsize=(10, 10))
    region_idx = (10, 105)
    voxel_idx = (100, 10500)
    typical_fr(res_path, voxel_idx, fig)
    typical_raster(res_path, region_idx, fig)
    heatmap_fc(res_path, fig)
    statics(res_path, fig)
    print("DONE ALL")
    plt.savefig(os.path.join(res_path, "total.png"), dpi=100)
    plt.close(fig)
