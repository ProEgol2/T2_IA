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

    result = []

    for i in range(11):
        threshold = i / 10
        eligible_indices = [j for j in range(len(P)) if ((j+1) / len(P)) >= threshold]
        if eligible_indices:
            max_val = max(P[j] for j in eligible_indices)
        else:
            max_val = P[-1]
        result.append(max_val)
    interpolated_precision.append(result)
    #print(result)

avg_interpolated_precision = (np.mean(np.array(interpolated_precision), axis=0))
r = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

plot = plt.plot(r, avg_interpolated_precision)

plt.title(f"Precision-Recall Curve\n{MODEL}, {DATASET}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.yticks(np.arange(0, 1.01, 0.2))
plt.yticks(np.arange(0, 1.01, 0.2))
plt.grid(True)
plt.savefig(f"P-C_{MODEL}_{DATASET}.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#print(avg_interpolated_precision)

MAP = np.mean(APs)
print(MAP)