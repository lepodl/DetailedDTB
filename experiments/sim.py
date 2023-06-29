# -*- coding: utf-8 -*- 
# @Time : 2023/3/13 10:10 
# @Author : lepold
# @File : simulation_singlecolumn.py

import sys

sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_develop/python")
sys.path.append("/public/home/ssct004t/project/zenglb/DetailedDTB/")
import os
import re
import subprocess as sp
import time
import numpy as np
import h5py

import torch

from model import simulation


class simulation_critical(simulation.simulation):
    @staticmethod
    def specified_sample_whole_brain(aal_region, neurons_per_population_base, specified_info=None,
                                             num_sample_voxel_per_region=1,
                                             num_neurons_per_voxel=300):
        """
        more convenient version to sample neurons for ``simulation`` object.

        neurons_per_population_base is read from :class:`population_base.npy <.TestBlock>` which is generated during generation of connection table.


        Parameters
        ----------

        aal_region: ndarrau
            indicate the brain regions label of each voxel.

        neurons_per_population_base: ndarray
            The accumulated number of neurons for each population , corresponding to the population_id.
            corresponding to [0, 1, 2, 3, 4,... 227029]

        num_sample_voxel_per_region: int, default=1
            the sample number of voxels in each region.

        num_neurons_per_voxel: int, default=300
            the sample number of neurons in each voxel .

        specified_info: ndarray
            according the specified_info info , we can randomly sample neurons which are from given voxel id.

        Returns
        -------

        ndarray which contain sample information

        ----------------|------------------
        0 th column     |     neuron id
        1 th column     |    voxel id
        2 th column     |    population id
        3 th column     |    region id
        ----------------|------------------

        """
        subcortical = np.array([361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374], dtype=np.int64)  # region index from 1
        subblk_base = [0]
        tmp = 0
        for i in range(len(aal_region)):
            if aal_region[i] in subcortical:
                subblk_base.append(tmp + 2)
                tmp = tmp + 2
            else:
                subblk_base.append(tmp + 8)
                tmp = tmp + 8
        subblk_base = np.array(subblk_base)
        uni_region = np.unique(np.sort(aal_region))
        num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
        sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)

        s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)

        if num_neurons_per_voxel == 6000:
            c1, c2, c3, c4, c5, c6, c7, c8 = 1600, 400, 1600, 400, 400, 200, 1200, 200  # sample_num=6000
        elif num_neurons_per_voxel == 200:
            c1, c2, c3, c4, c5, c6, c7, c8 = 54, 15, 56, 14, 13, 3, 37, 8  # sample_num=200
        elif num_neurons_per_voxel == 300:
            c1, c2, c3, c4, c5, c6, c7, c8 = 80, 20, 80, 20, 20, 10, 60, 10
        elif num_neurons_per_voxel == 900:
            c1, c2, c3, c4, c5, c6, c7, c8 = 240, 60, 240, 60, 60, 30, 180, 30
        else:
            raise "invalid sample_num"

        for num in range(len(uni_region)):
            i = uni_region[num]
            j = 0
            while True:
                # print("sampling for region: ", i)
                try:
                    if specified_info is None:
                        choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
                    else:
                        specified_info_index = np.where(specified_info[:, 3 == i])
                        choices = np.unique(specified_info[specified_info_index, 1])
                    count_voxel = num * num_sample_voxel_per_region
                    for choice in choices:
                        if i in subcortical:
                            id = choice * 10 + 6
                            neurons = np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1])
                            # print("len(neurons)", len(neurons), "s1", s1)
                            sample1 = np.random.choice(neurons, size=s1, replace=False)
                            id = choice * 10 + 7
                            neurons = np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1])
                            # print("len(neurons)", len(neurons), "s2", s2)
                            sample2 = np.random.choice(neurons, size=s2, replace=False)
                            sample = np.concatenate([sample1, sample2])
                            sub_blk = np.concatenate(
                                [np.ones_like(sample1) * (subblk_base[choice]),
                                 np.ones_like(sample2) * (subblk_base[choice] + 1)])[:,
                                      None]
                            sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
                            sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
                            sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1),
                            :] = sample
                        else:
                            sample_this_region = []
                            sub_blk = []
                            for yushu, size in zip(np.arange(2, 10), np.array([c1, c2, c3, c4, c5, c6, c7, c8])):
                                id = choice * 10 + yushu
                                neurons = np.arange(neurons_per_population_base[id],
                                                    neurons_per_population_base[id + 1])
                                # print("len(neurons)", len(neurons), "size", size)
                                sample1 = np.random.choice(neurons, size=size, replace=False)
                                sample_this_region.append(sample1)
                                sub_blk.append(np.ones(size) * (subblk_base[choice] + yushu))
                            sample_this_region = np.concatenate(sample_this_region)
                            sub_blk = np.concatenate(sub_blk)
                            sample = np.stack([sample_this_region, np.ones(num_neurons_per_voxel) * choice, sub_blk,
                                               np.ones(num_neurons_per_voxel) * i], axis=-1)
                            sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1),
                            :] = sample
                        count_voxel += 1
                    break
                except ValueError as e:
                    print('Exception {} {}'.format(type(e), e))
                    j = j + 1
                    if j < 50:
                        continue
                    else:
                        raise e
        return sample_idx.astype(np.int64)

    def __init__(self, ip: str, block_path: str, **kwargs):
        super().__init__(ip, block_path, **kwargs)

    def sample(self):
        global bock_dir
        brain_file = h5py.File("../data/raw_data/NSR_data_May22.mat", "r")
        region_label = np.array(brain_file['NSR_dti_HCPex_label']).squeeze()
        population_base = np.load(os.path.join(block_dir, "supplementary_info", "population_base.npy"))
        sample_idx = self.specified_sample_whole_brain(region_label, population_base, num_neurons_per_voxel=300)
        np.save(os.path.join(block_dir, "supplementary_info", "sample_dix.npy"), sample_idx)
        sample_idx = torch.from_numpy(sample_idx[:, 0]).cuda()
        self.block_model.set_samples(sample_idx)
        self.num_sample = sample_idx.shape[0]

        return self.num_sample


def open_server(slurm_name: str, nodes: int, single_slots: int = 4):
    f1 = open("/public/home/ssct004t/project/zenglb/DetailedDTB/experiments/server.slurm", 'r+', encoding='utf-8')
    content = f1.read()
    # print("Original current:" + '\n' + content)
    f1.close()
    content = re.sub(r"-J\s.*?\n", "-J %s\n" % slurm_name, content)
    content = re.sub(r"-N\s.*?\n", "-N %s\n" % nodes, content)
    content = re.sub(r"(==2.*?)=\d+", r"\1=%d" % single_slots, content)

    # print("After modify", '\n' + content)

    with open("/public/home/ssct004t/project/zenglb/DetailedDTB/experiments/server.slurm", "w",
              encoding='utf-8') as f2:
        f2.write(content)
        print("modify server.slurm！")

    command = "sbatch /public/home/ssct004t/project/zenglb/DetailedDTB/experiments/server.slurm 2>&1 | tr -cd '[0-9]'"
    job_id = sp.check_output(command, shell=True)
    job_id = str(job_id, "utf-8")
    print(f"server job id is {job_id}")

    echos = 0
    ip = None
    while echos < 10:
        try:
            with open(f"/public/home/ssct004t/project/zenglb/DetailedDTB/experiments/log/{job_id}.o", "r+",
                      encoding='utf-8') as out:
                lines = out.read()
            ip = re.findall(r"\d+\.\d+\.\d+\.\d+:\d+", lines)[0]
            break
        except:
            time.sleep(15)
            echos += 1
            continue
    print(f"server ip is {ip}")
    return ip


block_dir = "/public/home/ssct004t/project/zenglb/DetailedDTB/data/table_file/dti_distribution_200m_whole_brain_bounding_new_outer1dot9"
write_path = "/public/home/ssct004t/project/zenglb/DetailedDTB/data/result_data/simulation_june27_0.0025_0.0098"
block_path = os.path.join(block_dir, "module/uint8")
ip = open_server("nsr", nodes=6, single_slots=4)
assert ip is not None
model = simulation_critical(ip, block_path, dt=0.1, route_path=None, column=True, print_info=False,
                            vmean_option=False, imean_option=False,
                            sample_option=True, name="spike_dynamics_0.1_normal", write_path=write_path, draw_figs=True)
model.sample()
model.update(10, 0.0025)
model.update(11, 0.)
model.update(12, 0.0098)
model.update(13, 0.)
model.block_model.update_ou_background_stimuli(4, 0.40, 0.15)
model(step=800, observation_time=10, hp_index=None, hp_path=None)
