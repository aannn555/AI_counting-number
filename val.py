#!/usr/bin/env python
# coding: utf-8

# In[42]:


import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import scipy.io as sio
#get_ipython().magic(u'matplotlib inline')


# In[10]:


from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


# In[3]:


root = 'ShanghaiTech_Crowd_Counting_Dataset/'


# In[4]:


#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_test]


# In[5]:


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


# In[6]:


model = CSRNet()


# In[7]:


model = model.cuda()


# In[38]:


checkpoint = torch.load('0model_best.pth.tar')


# In[39]:


model.load_state_dict(checkpoint['state_dict'])


# In[45]:


mae = 0
mae2=0
for i in range(len(img_paths)):
    mae2=0
    img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    img[0,:,:]=img[0,:,:]-92.8207477031
    img[1,:,:]=img[1,:,:]-95.2757037428
    img[2,:,:]=img[2,:,:]-104.877445883
    img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    mae2= abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    plt.imshow(np.squeeze(np.asarray(output.detach().cpu())),cmap=CM.jet)
    print(img_paths[i].split('/')[4].split('.')[0])
    plt.savefig('save/save_'+img_paths[i].split('/')[4].split('.')[0])
    print("mae:",mae2)
    print("predict:",output.detach().cpu().sum().numpy())
    print("groundtruth:",np.sum(groundtruth))
    x=np.asarray([mae2,output.detach().cpu().sum().numpy(),np.sum(groundtruth)])
    sio.savemat('matlab/savedata'+img_paths[i].split('/')[4].split('.')[0]+'.mat', {'x': x})
print ("mae/len:",mae/len(img_paths))
y=np.asarray([mae/len(img_paths)])
sio.savemat('matlab/overall.mat', {'y': y})
