# -*- coding: utf-8 -*-
# @Time : 2022/8/7 13:07
# @Author : lepold
# @File : simulation.py


import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from cuda_develop.python.dist_blockwrapper_pytorch import BlockWrapper as block

from model.bold_model_pytorch import BOLD
from utils.default_params import bold_params, v_th
from utils.helpers import torch_2_numpy
from utils.pretty_print import pretty_print, table_print


class simulation(object):
    """
    By giving the IP and block path, a simulation object is initialized.
    Usually in the case: a one ensemble, i.e., one simulation ensemble.

    default params can be found in *params.py*,
    This is for micro-column version or voxel version.

    Specifically, for micro-column version, the overlap=10 and voxel version 1.
    It means that in DA procedure, the step units in indexing the populations between ensembles.

    Parameters
    ----------

    block_path: str
        the dir which saves the block.npz

    ip: str
        the server listening address

    route_path : str or None, default is None
        the routing results.

    kwargs: other positional params which are specified.

    """

    @staticmethod
    def merge_data(a, weights=None, bins=10, range=None, ):
        """
        merge a dataset to a desired shape. such as merge different populations to
        a voxel to calculate its size.

        Parameters
        ----------
        a : ndarray
            Input data. It must be a flattened array.

        weights: ndarray
            An array of weights, of the same shape as `a`

        bins : int
            such as the number of voxels in this simulation object

        range: (float, float)
            The lower and upper range of the bins, for micro-column,
            its upper range should be divided by 10, and for voxel should be divided by 2.

        Returns
        ----------
        merged data : array
            The values of the merged result.

        """
        return np.histogram(a, weights=weights, bins=bins, range=range)[0]

    def __init__(self, ip: str, block_path: str, dt: float = 1., route_path=None, column=True, **kwargs):
        if column:
            self.block_model = block(ip, block_path, dt, route_path=route_path, overlap=10)
            self.populations_per_voxel = 10
        else:
            self.block_model = block(ip, block_path, dt, route_path=route_path, overlap=2)
            self.populations_per_voxel = 2
        self.bold = BOLD(**bold_params)
        self.population_id = self.block_model.subblk_id
        self.population_id_cpu = self.population_id.cpu().numpy()

        if column:
            self.num_voxels = int((self.population_id_cpu.max() + 9) // 10)
        else:
            self.num_voxels = int((self.population_id_cpu.max() + 1) // 2)
        self.num_populations = int(self.block_model.total_subblks)
        self.num_neurons = int(self.block_model.total_neurons)

        self.num_neurons_per_population = self.block_model.neurons_per_subblk
        self.num_neurons_per_population_cpu = self.num_neurons_per_population.cpu().numpy()
        self.num_neurons_per_voxel_cpu = self.merge_data(self.population_id_cpu, self.num_neurons_per_population_cpu,
                                                         self.num_voxels,
                                                         (0, self.num_voxels * self.populations_per_voxel))

        self.write_path = kwargs.get('write_path', "./")
        os.makedirs(self.write_path, exist_ok=True)
        self.name = kwargs.get('name', "simulation")
        self.print_info = kwargs.get("print_info", False)
        self.vmean_option = kwargs.get("vmean_option", False)
        self.sample_option = kwargs.get("sample_option", False)
        self.imean_option = kwargs.get("imean_option", False)
        self.draw_figs = kwargs.get("draw_figs", False)

    def clear_mode(self):
        """
        Make this simulation to a clear state with no printing and only default freqs and bold as the output info.
        Returns
        -------

        """
        self.sample_option = False
        self.vmean_option = False
        self.imean_option = False
        self.print_info = False
        self.draw_figs = False

    def update(self, hp_index, param):
        """
        In other words, we use the same value to update a certain set of parameters of neurons.

        Parameters
        ----------
        hp_index: int
            indicate the column index among 21 attributes of LIF neuron.

        param: float

        """
        print(f"update {hp_index}th attribute, to value {param}\n")
        population_info = torch.stack(
            torch.meshgrid(self.population_id, torch.tensor([hp_index], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        param = torch.ones(self.num_populations, device="cuda:0") * param
        self.block_model.assign_property_by_subblk(population_info, param)

    def sample(self, *args, **kwargs):
        """

        set sample idx of neurons in this simulation object.
        Due to the routing algorithm, we utilize the disorder population idx and neurons num.

        Parameters
        ----------
        aal_region: ndarray
            indicate the brain regions label of each voxel appeared in this simulation.

        population_base: ndarray
            indicate range of neurons in each population.

        """

        self.block_model.set_samples_by_specifying_popu_idx(self)
        num_sample = 150
        self.num_sample = num_sample

        return num_sample

    def mul_property_by_subblk(self, index, value):
        """
        In DA process, we use it to update parameters which are sampled from a distribution.
        Parameters
        ----------
        index: Tensor
            the index of neuronal attribute which is specified to update, shape=(n, 2).
        value: Tensor
            the value of hyperparamter corresponding to above attribute.
        Returns
        -------

        """
        assert isinstance(index, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        self.block_model.mul_property_by_subblk(index, value)

    def gamma_initialize(self, hp_index, population_id=None, alpha=5., beta=5.):
        """
        Use a distribution of gamma(alpha, beta) to update neuronal params,
        where alpha = beta = 5.

        Parameters
        ----------

        hp_index: int
            indicate the column index among 21 attributes of LIF neuron.

        population_id: torch.tensor, default=None
            index of population whose parameters follow gamma distribution

        alpha: float, default=5.

        beta: float, default=5.

        """

        population_id = self.population_id if population_id is None else population_id
        print(f"gamma_initialize {hp_index}th attribute, to value gamma({alpha}, {beta}) distribution\n")
        population_info = torch.stack(
            torch.meshgrid(population_id, torch.tensor([hp_index], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        alpha = torch.ones(len(population_id), device="cuda:0") * alpha
        beta = torch.ones(len(population_id), device="cuda:0") * beta
        out = self.block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)
        print(f"gamma_initializing {hp_index}th attribute is {out}")

    def evolve(self, step, vmean_option=False, sample_option=False, imean_option=False, bold_detail=False) -> tuple:
        """

        The block evolve one TR time, i.e, 800 ms as default setting.

        Parameters
        ----------
        step: int, default=800
            iter number in one observation time point.

        sample_option: bool, default=False
            if True, block.run will contain freqs statistics, Tensor: shape=(num_populations, ).

        vmean_option: bool, default=False
            if True, block.run will contain vmean statistics, Tensor: shape=(num_populations, ).

        imean_option: bool, default=False
            if True, block.run will contain imean statistics, Tensor: shape=(num_populations, ).

        Returns
        ----------
        freqs: torch.Tensor
            total freqs of different populations, shape=(step, num_populations).

        vmean: torch.Tensor
            average membrane potential of different populations, shape=(step, num_populations).

        sample_spike: torch.Tensor
            the spike of sample neurons, shape=(step, num_sample_voxels).

        sample_v: torch.Tensor
            the membrane potential of sample neurons, shape=(step, num_sample_voxels).

        vmean: torch.Tensor
            average of synaptic current of different populations, shape=(step, num_populations).

        bold_out: torch.Tensor
            the bold out of voxels at the last step in one observation time poitn, shape=(num_voxels).

            .. versionadded:: 1.0

        """
        total_res = []
        fr = []

        for return_info in self.block_model.run(step, freqs=True, freq_char=True, vmean=vmean_option, sample_for_show=sample_option, iou=True,
                                                imean=imean_option, t_steps=10, equal_sample=True):
            Freqs, *others = return_info
            total_res.append(return_info)

            freqs = Freqs.cpu().numpy()
            act = self.merge_data(self.population_id_cpu, freqs, self.num_voxels,
                                  (0, self.num_voxels * self.populations_per_voxel))
            act = (act / self.num_neurons_per_voxel_cpu).reshape(-1)
            act = torch.from_numpy(act).cuda()
            fr.append(act)
            bold_out = self.bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
        if bold_detail:
            return act, self.bold.s, self.bold.q, self.bold.v, self.bold.f_in, bold_out
        # temp_freq = torch.stack([x[0] for x in total_res], dim=0)
        temp_fr = torch.stack(fr, dim=0)
        out = (temp_fr,)
        out_base = 1
        if vmean_option:
            temp_vmean = torch.stack([x[out_base] for x in total_res], dim=0)
            out += (temp_vmean,)
            out_base += 1
        if sample_option:
            temp_spike = torch.stack([x[out_base] for x in total_res], dim=0)
            temp_vsample = torch.stack([x[out_base + 1] for x in total_res], dim=0)
            # temp_spike &= (torch.abs(temp_vsample - v_th) / 50 < 1e-5)
            out += (temp_spike, temp_vsample,)
            out_base += 2
        if imean_option:
            temp_imean = torch.stack([x[out_base] for x in total_res], dim=0)
            out += (temp_imean,)
        out += (bold_out,)
        return out

    def run(self, step=800, observation_time=100, hp_index=None, hp_total=None):
        """
        Run this block and save the returned block information.

        Parameters
        ----------
        step: int, default=800
             iter number in one observation time point.

        observation_time: int, default=100
            total time points, equal to the time points of bold signal.

        """

        if self.sample_option and not hasattr(self, 'num_sample'):
            raise NotImplementedError('Please set the sampling neurons first in simulation case')

        start_time = time.time()
        if hp_total is not None:
            state = "after"
            for k in hp_index:
                self.gamma_initialize(k, self.population_id)
            population_info = torch.stack(
                torch.meshgrid(self.population_id, torch.tensor(hp_index, dtype=torch.int64, device="cuda:0")),
                dim=-1).reshape((-1, 2))
            self.block_model.mul_property_by_subblk(population_info, hp_total[0].reshape(-1))
            hp_total = hp_total[1:, ]
            total_T = hp_total.shape[0]
            assert observation_time <= total_T
        else:
            state = "before"
            total_T = None

        freqs, _ = self.evolve(3200, vmean_option=False, sample_option=False, imean_option=False)
        Init_end = time.time()
        if self.print_info:
            print(
                f"mean fre: {torch.mean(torch.mean(freqs / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}")
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")

        pretty_print("Begin Simulation")
        print(f"Total time is {total_T}, we only simulate {observation_time}\n")

        bolds_out = np.zeros([observation_time, self.num_voxels], dtype=np.float32)
        for ii in range((observation_time - 1) // 50 + 1):
            nj = min(observation_time - ii * 50, 50)
            Firing = np.zeros([nj, step, self.num_voxels], dtype=np.float32)  # voxel firing rate
            if self.vmean_option:
                Vmean = np.zeros([nj, step, self.num_populations], dtype=np.float32)
            if self.sample_option:
                Spike = np.zeros([nj, step, self.num_sample], dtype=np.uint8)
                Vi = np.zeros([nj, step, self.num_sample], dtype=np.float32)
            if self.imean_option:
                Imean = np.zeros([nj, step, self.num_populations], dtype=np.float32)

            for j in range(nj):
                i = ii * 50 + j
                t_sim_start = time.time()
                if hp_total is not None:
                    self.block_model.mul_property_by_subblk(population_info, hp_total[i].reshape(-1))
                out = self.evolve(step, vmean_option=self.vmean_option, sample_option=self.sample_option,
                                  imean_option=self.imean_option)
                Firing[j] = torch_2_numpy(out[0])
                out_base = 1
                if self.vmean_option:
                    Vmean[j] = torch_2_numpy(out[out_base])
                    out_base += 1
                if self.sample_option:
                    Spike[j] = torch_2_numpy(out[out_base])
                    Vi[j] = torch_2_numpy(out[out_base + 1])
                    out_base += 2
                if self.imean_option:
                    Imean[j] = torch_2_numpy(out[out_base])

                bolds_out[i, :] = torch_2_numpy(out[-1])
                t_sim_end = time.time()
                print(
                    f"{i}th observation_time, mean fre: {Firing[j].mean():.4f}, cost time {t_sim_end - t_sim_start:.1f}")
                if self.print_info:
                    stat_data, stat_table = self.block_model.last_time_stat()
                    np.save(os.path.join(self.write_path, f"stat_{i}.npy"), stat_data)
                    stat_table.to_csv(os.path.join(self.write_path, f"stat_{i}.csv"))
                    print('print_stat_time', time.time() - t_sim_end, stat_table)
            if self.sample_option:
                np.save(os.path.join(self.write_path, f"spike_{state}_assim_{ii}.npy"), Spike)
                np.save(os.path.join(self.write_path, f"vi_{state}_assim_{ii}.npy"), Vi)
            if self.vmean_option:
                np.save(os.path.join(self.write_path, f"vmean_{state}_assim_{ii}.npy"), Vmean)
            if self.imean_option:
                np.save(os.path.join(self.write_path, f"imean_{state}_assim_{ii}.npy"), Imean)
            np.save(os.path.join(self.write_path, f"firing_{state}_assim_{ii}.npy"), Firing)
            np.save(os.path.join(self.write_path, f"bold_{state}_assim.npy"), bolds_out)
            # savemat(os.path.join(self.write_path, f"bold_{state}_assim.mat"), {'bolds_out': bolds_out})
            if self.draw_figs:
                fig = plt.figure(figsize=(6, 7))
                # print("Spike", Spike.shape)
                Spike = Spike[-2:, :, :].reshape((2 * step, -1))
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.75], frameon=True)
                spike_events = [Spike[:, i].nonzero()[0] for i in range(300)]
                s = 0
                colors = ["tab:blue", "tab:red"]
                total = 300
                sample_dis = np.array([80, 20, 80, 20, 20, 10, 60, 10], dtype=np.int32)
                names = ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
                for i, size in enumerate(sample_dis):
                    e = s + size
                    color = colors[i % 2]
                    y_inter = (s + e) / 2 / total - 0.02
                    fr = Spike[:, s:e].mean() * 1000
                    ax.eventplot(spike_events[s:e], lineoffsets=np.arange(s, e), colors=color, linestyles="dashed")
                    # xx, yy = Spike[:, s:e].nonzero()
                    # yy = yy + s
                    # ax.scatter(xx, yy, marker=',', s=1., color=color)
                    s = e
                    ax.text(0.8, y_inter, names[i] + f": {fr:.1f}Hz", color=color, fontsize=9, transform=ax.transAxes)
                    ax.set_ylim([0, 300])
                    ax.set_yticks([0, 300])
                    ax.set_yticklabels([0, 300])
                    ax.set_ylabel("Neuron")
                    ax.set_xlim([0, 1600])
                    ax.set_xticks([0, 1000, 1600])
                    ax.set_xticklabels(["0.0", "1.0", "1.6"])
                    ax.set_xlabel("Time (s)")
                    # ax.invert_yaxis()
                fig.savefig(os.path.join(self.write_path, "column.png"))
                plt.close(fig)
                fig_new, ax = plt.subplots(3, 2, figsize=(8, 7))
                for i in range(3):
                    for j in range(2):
                        sample_voxel = np.random.randint(0, self.num_voxels, 1)
                        bold = bolds_out[:, sample_voxel]
                        ax[i, j].plot(np.arange(bold.shape[0]), bold, lw=1.)
                        ax[i, j].set_ylim([0., 0.04])
                        ax[i, j].spines["top"].set_visible(False)
                        ax[i, j].spines["right"].set_visible(False)
                        ax[i, j].set_xlabel("Time")
                        ax[i, j].set_ylabel("bold")
                fig_new.tight_layout()
                fig_new.savefig(os.path.join(self.write_path, "bold.png"))
                plt.close(fig_new)

        pretty_print(f"Totally have Done, Cost time {time.time() - start_time:.2f} ")

    def __call__(self, step=800, observation_time=100, hp_index=None, hp_path=None):
        """
        Give this objects the ability to behave like a function, to simulate the LIF nn.

        Parameters
        ----------

        hp_index: list, default is None.

        hp_path: str, default is None.


        Notes
        ---------

        if hp info is None, it equally says that we don't need to update neuronal attributes
        during the simulation period.

        """

        info = {'name': self.name, "num_neurons": self.num_neurons, 'num_voxel': self.num_voxels,
                'num_populations': self.num_populations, 'step': step}
        table_print(info)
        if hp_path is not None:
            hp_total = np.load(hp_path)
            assert hp_total.shape[1] == self.num_populations
            hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
            self.run(step=step, observation_time=observation_time, hp_index=hp_index, hp_total=hp_total)
        else:
            self.run(step=step, observation_time=observation_time, hp_index=None, hp_total=None)

        self.block_model.shutdown()
