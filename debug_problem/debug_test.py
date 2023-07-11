# -*- coding: utf-8 -*- 
# @Time : 2023/6/29 14:30 
# @Author : lepold
# @File : debug_test.py


from cuda_develop.python.dist_blockwrapper_pytorch import BlockWrapper
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Model simulation")
parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
args = parser.parse_args()


path = '/public/home/ssct004t/project/Digital_twin_brain/data/2000_block/dtb_param4to1'
route_path = None
ip = args.ip
delta_t = 1.
t_steps = 1

dist_block = BlockWrapper(ip, os.path.join(path, 'uint8'), delta_t, route_path=route_path)

N = 50
sample_idx = torch.arange(N, dtype=torch.int64).cuda()
problem_iteration = 20971 * 800   # start=20000 * 800 ; end=21000 * 800
start=20000 * 800
end = 21000 * 800
document = 1000 * 800
run_number = 21000

dist_block.update_ou_background_stimuli(10., 0.66, 0.12)

dist_block.set_samples(sample_idx)

params = [0., 0., 0., 0.]

for i, ind in enumerate(range(10, 14)):
    population_info = torch.stack(
        torch.meshgrid(dist_block.subblk_id, torch.tensor([ind], dtype=torch.int64, device="cuda:0")),
        dim=-1).reshape((-1, 2))
    param = torch.ones(dist_block.total_subblks, device="cuda:0") * params[i]
    dist_block.assign_property_by_subblk(population_info, param)

print('\nsample_idx.size()', sample_idx.size())
print("\ntotal subblks", dist_block.subblk_id.shape[0])
print("\n")

sample_spike = torch.empty([document, sample_idx.shape[0]], dtype=torch.uint8)
sample_vi = torch.empty([document, sample_idx.shape[0]], dtype=torch.float32)
sample_freqs = torch.empty([run_number * 800, dist_block.subblk_id.shape[0]], dtype=torch.int32)
sample_ou = torch.empty([document, sample_idx.shape[0]], dtype=torch.float32)
sample_syn_current = torch.empty([document, t_steps, sample_idx.shape[0]], dtype=torch.float32)

for j in range(run_number):
    for idx, (freqs, spike, vi, i_sy, iouu) in enumerate(
            dist_block.run(800, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=True,
                           t_steps=t_steps, equal_sample=True)):
        temp_count = j * 800 + idx
        sample_freqs[temp_count, :] = freqs.cpu()
        if start <= temp_count < end:
            sample_spike[temp_count-start, :] = spike.cpu()
            sample_vi[temp_count-start, :] = vi.cpu()
            sample_ou[temp_count-start, :] = iouu.cpu()
            sample_syn_current[temp_count-start, :] = i_sy.cpu()
    print('mean firing rate in iteration', j,
          torch.div(torch.sum(freqs.float(), dim=0) * 1000 / delta_t, dist_block.neurons_per_subblk.float()))
sample_spike = sample_spike.cpu().numpy()
sample_vi = sample_vi.cpu().numpy()
sample_syn_current = sample_syn_current.numpy()
dist_block.shutdown()

sample_freqs = sample_freqs.numpy()
sample_spike = sample_spike.numpy()
sample_vi = sample_vi.numpy()
sample_ou = sample_ou.numpy()
sample_syn_current = sample_syn_current.numpy()


fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
ax.plot(sample_freqs.sum(axis=-1), lw=1.)
ax.set_xlabel("iteration")
ax.set_yabel("freqs")
fig.savefig("./debug.png")

np.save("./freqs.npy", sample_freqs)
np.save("./spike.npy", sample_spike)
np.save("./vi.npy", sample_vi)
np.save("./ou.npy", sample_ou)
np.save("./syn_current.npy", sample_syn_current)
print("\n\nDone")






