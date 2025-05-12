from torchvision import transforms, models
from PIL import Image
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
import os

DATASET = 'simple1k'
MODEL = 'resnet18'

data_dir = 'simple1k'
image_dir = os.path.join(data_dir, 'images')
list_of_images = os.path.join(data_dir, 'list_of_images.txt')

feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))

if DATASET == 'Paris_val':
    with open(list_of_images, "r+") as file: 
        files = [[f.split('\t')[0].split("/")[1], f.split('\t')[1]] for f in file]
else:
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        # [[im_path, label], ...]

feats = np.load(feat_file)    
norm2 = np.linalg.norm(feats, ord = 2, axis = 1,  keepdims = True)
feats_n = feats / norm2
sim = feats_n @ np.transpose(feats_n)
sim_idx = np.argsort(-sim, axis = 1)

APs = []
interpolated_precision = []

for idx1 in range(len(sim)):
    P = []
    count = 0
    for idx2, j in enumerate(sim_idx[idx1, 1:]):
        if files[j][1].strip() == files[idx1][1].strip():
            count += 1
            P.append(count/(idx2+1))
    APs.append(sum(P)/count if count != 0 else 0)

APs = np.array(APs)

#five_idx = np.argsort(-APs)[:5] For 5 best Aps
five_idx = np.argsort(APs)[:5]

fig, ax = plt.subplots(5,10, figsize=(22, 10))
ax = ax.flatten()
count = 0

for i, idx in enumerate(five_idx):
    filename = os.path.join(image_dir, files[idx][0])
    im = io.imread(filename)
    im = transform.resize(im, (64,64)) 
    ax[count].imshow(im)                 
    ax[count].set_axis_off()
    ax[count].set_title(f"{files[idx][1]}, AP = {APs[idx]}", fontsize = 8)
    count += 1
    for j, idx2 in enumerate(sim_idx[idx, 1:10]):
        filename = os.path.join(image_dir, files[idx2][0])
        im = io.imread(filename)
        im = transform.resize(im, (64,64)) 
        ax[count].imshow(im)                 
        ax[count].set_axis_off()
        ax[count].set_title(files[idx2][1], fontsize = 8)
        count += 1
ax[0].patch.set(lw=6, ec='b')
ax[0].set_axis_off()
plt.subplots_adjust(hspace=0.8)
plt.savefig(f"5_worst_{MODEL}_{DATASET}.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#MAP = np.mean(APs)
#print(MAP)

