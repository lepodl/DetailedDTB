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

import torch

from model import simulation


class simulation_critical(simulation.simulation):
    def __init__(self, ip: str, block_path: str, **kwargs):
        super().__init__(ip, block_path, **kwargs)

    def sample(self):
        population_id = torch.tensor([1002, 1003, 1004, 10005, 10006, 10007, 1008, 1009], dtype=torch.int64).cuda()
        sample_number = torch.tensor([80, 20, 80, 20, 20, 10, 60, 10], dtype=torch.int64).cuda() * 2
        self.block_model.set_samples_by_specifying_popu_idx(population_id, sample_number)
        self.num_sample = torch.sum(sample_number).item()

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


block_dir = "/public/home/ssct004t/project/zenglb/DetailedDTB/data/table_file/dti_distribution_500m_column_ordinary_5june/module"
write_path = "/public/home/ssct004t/project/zenglb/DetailedDTB/data/result_data/simulation_june5th"
block_path = os.path.join(block_dir, "uint8")
ip = open_server("nsr", nodes=14, single_slots=2)
assert ip is not None
model = simulation_critical(ip, block_path, dt=1., route_path=None, column=True, print_info=False,
                            vmean_option=False, imean_option=True,
                            sample_option=True, name="spike_dynamics", write_path=write_path, draw_figs=True)
model.sample()
model.update(10, 0.0014)
model.update(11, 0.00005)
model.update(12, 0.0009)
model.block_model.update_ou_background_stimuli(4, 0.40, 0.15)
model(step=800, observation_time=50, hp_index=None, hp_path=None)
