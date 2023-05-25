# -*- coding: utf-8 -*- 
# @Time : 2023/5/21 13:54 
# @Author : lepold
# @File : preprocess_original.py
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import sparse
from scipy.io import loadmat, savemat

file = h5py.File(
    "/public/home/ssct004t/project/chao_data/fmri_prep/A1_fMRIPrep_DTI_voxel_structure_data_jianfeng_Connectome_ye_info2.mat",
    "r")
file.keys()

aal_label = file['dti_aal_label'][:].squeeze()
N = len(aal_label)
aal_label = aal_label.astype(np.uint8)
brain_part = file['dti_brainPart_label'][:].squeeze()
gm = file["dti_grey_matter"][:].squeeze()
brain_part[brain_part == 230] = 205
nii_label = file["dti_label_num"][:].squeeze()
xyz = file["dti_xyz"][:]
rest_bold = file["dti_rest_state"][:].squeeze()
task_bold_visual = loadmat(
    "/public/home/ssct004t/project/chao_data/fmri_prep/A2_1_DTI_voxel_structure_data_jianfeng_Connectome_v2_task.mat")[
    'evaluation_run1'].T
task_bold_auditory = loadmat(
    "/public/home/ssct004t/project/chao_data/fmri_prep/A2_1_DTI_voxel_structure_data_jianfeng_Connectome_v2_task.mat")[
    'evaluation_run1_auditory'].T
assert task_bold_visual.shape[0] == len(gm)
assert task_bold_auditory.shape[0] == len(gm)

reader = pd.read_csv(
    "/public/home/ssct004t/project/chao_data/fmri_prep/whole_connectome_dti_ye.csv",
    sep="\t", chunksize=1000, header=None)
conn_prob = np.zeros((N, N), dtype=np.int64)
for i, chunk in enumerate(reader):
    chunk_source = chunk.values.astype(np.int32)
    print(i, chunk_source.shape)
    conn_prob[i * 1000:(i + 1) * 1000, :] = chunk_source
type(conn_prob)

index = np.argsort(brain_part)
conn_prob = conn_prob[index]
conn_prob = conn_prob[:, index]
aal_label = aal_label[index]
brain_part = brain_part[index]
gm = gm[index]
rest_bold = rest_bold[index]
task_bold_visual = task_bold_visual[index]
task_bold_auditory = task_bold_auditory[index]
nii_label = nii_label[index]
xyz = xyz[index]

exclude_index = np.where(gm > 0.3)[0]
gm = gm[exclude_index]
brain_part = brain_part[exclude_index]
aal_label = aal_label[exclude_index]
rest_bold = rest_bold[exclude_index]
task_bold_visual = task_bold_visual[exclude_index]
task_bold_auditory = task_bold_auditory[exclude_index]
nii_label = nii_label[exclude_index]
xyz = xyz[exclude_index]
# gm /= gm.sum()
conn_prob = conn_prob[exclude_index]
conn_prob = conn_prob[:, exclude_index]

conn_prob[conn_prob <= 1] = 0  # for sparsity
exclude_index = np.where(np.sum(conn_prob, axis=1) != 0)[0]
print(len(exclude_index))
conn_prob = conn_prob[exclude_index]
conn_prob = conn_prob[:, exclude_index]
assert np.sum(conn_prob.sum(axis=1) > 0) == conn_prob.shape[0]
# conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
gm = gm[exclude_index]
aal_label = aal_label[exclude_index]
brain_part = brain_part[exclude_index]
task_bold_visual = task_bold_visual[exclude_index]
task_bold_auditory = task_bold_auditory[exclude_index]
nii_label = nii_label[exclude_index]
xyz = xyz[exclude_index]
rest_bold = rest_bold[exclude_index]

divide_point = (brain_part < 205).sum()
# degree_partition = (brain_part <= 205).sum()

index = np.where(brain_part==200)[0]
conn_prob = conn_prob[index]
conn_prob = conn_prob[:, index]
aal_label = aal_label[index]
brain_part = brain_part[index]
gm = gm[index]
rest_bold = rest_bold[index]
task_bold_visual = task_bold_visual[index]
task_bold_auditory = task_bold_auditory[index]
nii_label = nii_label[index]
xyz = xyz[index]


brain_parts = []
for id in brain_part:
    if id == 200.:
        brain_parts.append("cortex")
    elif id == 210.:
        brain_parts.append("brainstem")
    elif id == 220.:
        brain_parts.append("cerebellum")
    elif id == 205:
        brain_parts.append("subcortex")
    else:
        raise NotImplementedError

assert (np.array(brain_parts) == "cortex").all()

conn_prob = sparse.COO(conn_prob)
print("density", conn_prob.density)
N = conn_prob.shape[0]
assert divide_point == N
assert gm.shape[0] == nii_label.shape[0] == N
brain_file = {"conn_prob": conn_prob, "gm": gm, "aal_label": aal_label, "brain_parts": brain_parts,
              "rest_bold": rest_bold, "task_bold_visual": task_bold_visual, "task_bold_auditory": task_bold_auditory, "nii": nii_label,
              "xyz": xyz}
print("total voxels", gm.shape[0])
os.makedirs("/public/home/ssct004t/project/zenglb/DetailedDTB/data/raw_data/", exist_ok=True)
with open(
        "/public/home/ssct004t/project/zenglb/DetailedDTB/data/raw_data/cortex.pickle",
        "wb") as f:
    pickle.dump(brain_file, f)
print("Done")
