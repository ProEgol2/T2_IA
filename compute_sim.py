from torchvision import transforms, models
from PIL import Image
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
import os

DATASET = 'Paris_val'
MODEL = 'dinov2'

data_dir = DATASET
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

#--- compute similarity
feats = np.load(feat_file)    
norm2 = np.linalg.norm(feats, ord = 2, axis = 1,  keepdims = True)
feats_n = feats / norm2
sim = feats_n @ np.transpose(feats_n)
sim_idx = np.argsort(-sim, axis = 1)

#---- An example of results just pickin a random query
# the first image appearing must be the same as the query
query = np.random.permutation(sim.shape[0])[0]
k = 10
best_idx = sim_idx[query, :k+1]
print(sim[query, best_idx])

fig, ax = plt.subplots(1,11)
w = 0
for i, idx in enumerate(best_idx):        
    filename = os.path.join(image_dir, files[idx][0])
    im = io.imread(filename)
    im = transform.resize(im, (64,64)) 
    ax[i].imshow(im)                 
    ax[i].set_axis_off()
    ax[i].set_title(files[idx][1])
        
ax[0].patch.set(lw=6, ec='b')
ax[0].set_axis_on()            
plt.show()