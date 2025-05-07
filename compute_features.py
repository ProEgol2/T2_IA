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

if DATASET == 'Paris_val':
    with open(list_of_images, "r+") as file: 
        files = [[f.split('\t')[0].split("/")[1], f.split('\t')[1]] for f in file]
else:
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        # [[im_path, label], ...]
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
    ])

#Load the model     
model = None
if MODEL == 'resnet18' :
    model = models.resnet18(pretrained=True).to(device)
    model.fc = torch.nn.Identity() 
    dim = 512
if MODEL == 'resnet34' :
    model = models.resnet34(pretrained=True).to(device)
    model.fc = torch.nn.Identity() 
    dim = 512
if MODEL == 'clip' :
    model, preprocess_clip = clip.load("ViT-B/32", device=device)
    model = model.encode_image
    dim = 512
if MODEL == 'dinov2' :
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dim = 384

#Get features
with torch.no_grad():        
    n_images = len(files)
    features = np.zeros((n_images, dim), dtype = np.float32)

    for i, file in enumerate(files) :                
        filename = os.path.join(image_dir, file[0])
        image = Image.open(filename).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
        features[i,:] = model(image).cpu()[0,:]
        if i%100 == 0 :
            print('{}/{}'.format(i, n_images))            
    
    os.makedirs('data', exist_ok=True)
    feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
    np.save(feat_file, features)
    print('saving data ok')