# -*- coding: utf-8 -*- 
# @Time : 2022/8/10 14:31 
# @Author : lepold
# @File : test_generation.py
import os.path
import unittest

import psutil
from mpi4py import MPI

from generation.make_block import *


class TestBlock(unittest.TestCase):
    @staticmethod
    def print_system_info(card: int):
        mem = psutil.virtual_memory()
        # 系统总计内存
        zj = float(mem.total) / 1024 / 1024 / 1024
        # 系统已经使用内存
        ysy = float(mem.used) / 1024 / 1024 / 1024

        # 系统空闲内存
        kx = float(mem.free) / 1024 / 1024 / 1024

        # print('%4d卡 系统总计内存:%d.3GB' % (card, zj))
        # print('%4d卡 系统已经使用内存:%d.3GB' % (card, ysy))
        print('%4d卡 系统空闲内存:%d.3GB' % (card, kx))

    @staticmethod
    def _make_directory_tree(root_path, scale, extra_info, dtype="single"):
        """
        make directory tree for each subject.

        Parameters
        ----------
        root_path: str
            each subject has a root path.

        scale: int
            number of neurons of whole brain.
        degree:
            in-degree of each neuron.

        init_min: float
            the lower bound of uniform distribution where w is sampled from.

        init_max: float
            the upper bound of uniform distribution where w is sampled from.

        extra_info: str
            supplementary information.

        Returns
        ----------
        second_path: str
            second path to save connection table

        """
        os.makedirs(root_path, exist_ok=True)
        os.makedirs(os.path.join(root_path, "raw_data"), exist_ok=True)
        second_path = os.path.join(root_path,
                                   f"dti_distribution_{int(scale // 1e6)}_{extra_info}")
        os.makedirs(second_path, exist_ok=True)
        os.makedirs(os.path.join(second_path, "module"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "multi_module", dtype), exist_ok=True)  # 'single' means the precision.
        os.makedirs(os.path.join(second_path, "supplementary_info"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "DA"), exist_ok=True)

        return second_path, os.path.join(second_path, "module")

    @staticmethod
    def _add_laminar_cortex_model(conn_prob, gm):
        """
        Process the connection probability matrix, grey matter and degree scale for DTB with pure voxel and micro-column
        structure.  Each voxel is split into 2 populations (E and I). Each micro-column is spilt into 10 populations
        (L1E, L1I, L2/3E, L2/3I, L4E, L4I, L5E, L5I, L6E, L6I).

        Parameters
        ----------
        conn_prob: numpy.ndarray, shape [N, N]
            the connectivity probability matrix between N voxels/micro-columns.

        gm: numpy.ndarray, shape [N]
            the normalized grey matter in each voxel/micro-column.

        canonical_voxel: bool
            Ture for voxel structure; False for micro-column structure.

        Returns
        -------
        out_conn_prob: numpy.ndarray
            connectivity probability matrix between populations (shape [2*N, 2*N] for voxel; shape[10*N, 10*N] for micro
            -column) in the sparse matrix form.

        out_gm: numpy.ndarray
            grey matter for populations in DTB (shape [2*N] for voxel; shape[10*N] for micro-column).

        out_degree_scale: numpy.ndarray
            scale of degree for populations in DTB (shape [2*N] for voxel; shape[10*N] for micro-column).

        """
        lcm_connect_prob = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 3554, 804, 881, 45, 431, 0, 136, 0, 1020],
                                     [0, 0, 1778, 532, 456, 29, 217, 0, 69, 0, 396],
                                     [0, 0, 417, 84, 1070, 690, 79, 93, 1686, 0, 1489],
                                     [0, 0, 168, 41, 628, 538, 36, 0, 1028, 0, 790],
                                     [0, 0, 2550, 176, 765, 99, 621, 596, 363, 7, 1591],
                                     [0, 0, 1357, 76, 380, 32, 375, 403, 129, 0, 214],
                                     [0, 0, 643, 46, 549, 196, 327, 126, 925, 597, 2609],
                                     [0, 0, 80, 8, 92, 3, 159, 11, 76, 499, 1794]], dtype=np.float64
                                    )

        lcm_gm = np.array([0, 0,
                           33.8 * 78, 33.8 * 22,
                           34.9 * 80, 34.9 * 20,
                           7.6 * 82, 7.6 * 18,
                           22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
        lcm_gm = lcm_gm / lcm_gm.sum()
        weight = lcm_gm[::2]
        weight = weight / weight.sum(axis=0)
        weight = np.broadcast_to(weight, (conn_prob.data.shape[0], 5))

        lcm_connect_prob = lcm_connect_prob / np.sum(lcm_connect_prob)
        inner_conn = conn_prob.sum(axis=1).data * lcm_connect_prob[:, :10].sum()

        corrds1 = np.empty(
            [4, conn_prob.coords.shape[1] * lcm_connect_prob.shape[0] * int(lcm_connect_prob.shape[0] / 2)],
            dtype=np.int64)
        corrds1[3, :] = np.tile(np.repeat(np.arange(0, lcm_connect_prob.shape[0], 2), lcm_connect_prob.shape[0]),
                                conn_prob.coords.shape[1]).reshape([1, -1])

        corrds1[(0, 2), :] = np.broadcast_to(conn_prob.coords[:, :, None],
                                             [2, conn_prob.coords.shape[1],
                                              lcm_connect_prob.shape[0] * int(lcm_connect_prob.shape[0] / 2)]).reshape(
            [2, -1])
        corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                          [conn_prob.coords.shape[1] * int(lcm_connect_prob.shape[0] / 2),
                                           lcm_connect_prob.shape[0]]).reshape([1, -1])

        data1 = conn_prob.data[:, None] * lcm_connect_prob[:, -1]
        data1 = (data1[:, None, :] * weight[:, :, None]).reshape([-1])

        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, conn_prob.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(np.arange(conn_prob.shape[0], dtype=np.int64)[:, None],
                                        [conn_prob.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.reshape(lcm_connect_prob_inner.data[None, :] * inner_conn[:, None], [-1])

        coords = np.concatenate([corrds1, corrds2, ], axis=1)
        data = np.concatenate([data1, data2], axis=0)
        shape = [conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                 lcm_connect_prob.shape[1] - 1]

        index = np.where(data)[0]
        print(f"process zero value in conn_prob {len(data)}-->{len(index)}")
        coords = coords[:, index]
        data = data[index]

        out_conn_prob = sparse.COO(coords=coords, data=data, shape=shape)
        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        out_degree_scale = out_conn_prob.sum(axis=1).todense()
        out_degree_scale = out_degree_scale / out_degree_scale.sum()

        gm /= gm.sum()
        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape([-1])

        return out_conn_prob, out_gm, out_degree_scale

    def _test_generate_cortex_brain(self, root_path="table_file", degree=100,
                                    scale=int(1e9), dtype="uint8"):
        first_path, second_path = self._make_directory_tree(root_path, scale, "column_src", dtype=dtype)
        blocks = 200
        print(f"Total {scale} neurons for DTB, merge to {blocks} blocks")
        with open("raw_data/cortex.pickle", "rb") as f:
            file = pickle.load(f)
        conn_prob = file["conn_prob"]
        block_size = file["gm"]

        conn_prob, block_size, degree_scale = self._add_laminar_cortex_model(conn_prob, block_size)
        degree_ = (degree * degree_scale / block_size).astype(np.uint16)
        # print("original degree nonzero", np.where(degree_scale)[0].sum() / len(degree_scale))
        # print("actual degree nonzero", np.where(degree_)[0].sum() / len(degree_scale))
        # import matplotlib.pyplot as plt
        # plt.hist(degree_, bins=50, rwidth=0.8)
        # plt.savefig("./degree_.png")
        # plt.close()

        kwords = [{"V_th": -50,
                   "V_reset": -55,
                   'V_L': -70,
                   'C': 0.5,
                   'T_ref': 2,
                   'g_Li': 25 * 1e-3,
                   'g_ui': np.array([0.5, 0.1, 6., 0.]) * 1e-3,
                   'tao_ui': (2, 40, 20, 1),
                   'V_ui': (0, 0, -70, 0),
                   'noise_rate': 0.,
                   "size": int(max(b * scale, 1)) if b != 0 else 0}
                  for i, b in enumerate(block_size)]

        conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree_, dtype=dtype,
                                              )
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            if rank == 0:
                population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
                population_base = np.add.accumulate(population_base)
                population_base = np.insert(population_base, 0, 0)
                np.save(os.path.join(first_path, "supplementary_info", "population_base.npy"), population_base)
            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           avg_degree=tuple([degree + 20] * blocks),
                                           dtype=dtype,
                                           debug_block_dir=None,
                                           only_load=(i != 0))


if __name__ == "__main__":
    unittest.main()
