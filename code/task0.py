import _init_paths
import pdb
from datasets.factory import get_imdb
import visdom
import numpy as np
import cv2
import os
import subprocess

def get_DNS():
    sub = subprocess.Popen(['curl', '-s' ,'http://169.254.169.254/latest/meta-data/public-hostname'], stdout=subprocess.PIPE)
    dns = sub.stdout.read()
    return dns

address = 'http://'+get_DNS()
vis = visdom.Visdom(server=address, port=9040)

# vis = visdom.Visdom(server='http://ec2-52-14-238-23.us-east-2.compute.amazonaws.com', port='6006')

# imdb = get_imdb('voc_2007_trainval')

# idx = 2018

# image_set_index = imdb._load_image_set_index()
# image_index = image_set_index[idx]
# pascal_annotations = imdb._load_pascal_annotation(image_index)

# # Ground truth classes
# gt_classes = pascal_annotations['gt_classes']
# print('Images in '+image_index+ ' are:')
# for i in gt_classes:
#         print(imdb._classes[i-1]) #idx here starts from 1

# # Image path
# print('Image path for '+image_index+' is :')
# impath = imdb.image_path_at(idx)
# print(impath)


# #Visualizing the proposed bounding boxes
# gt_roi = imdb.gt_roidb()
# rpn_roi = imdb._load_rpn_roidb(gt_roi)
# rpn_bbox = rpn_roi['boxes']
# img = cv2.imread(impath)
# for b in rpn_bbox[:10]:
# 	cv2.rectangle(img, tuple(b[0:2]), tuple(b[2:4]), (0,0,204),2)
# img = np.transpose(img, axes=(2,0,1))
# vis.image(img, 'Top 10 bounding box proposals')

# #Visualizing GT bounding boxes
# gt_bbox = pascal_annotations['boxes']
# img = cv2.imread(impath)
# imorig = img
# for b in gt_bbox[:10]:
#         cv2.rectangle(img, tuple(b[0:2]), tuple(b[2:4]), (0,204,0),2)
# img = np.transpose(img, axes=(2,0,1))
# # ipdb.set_trace()
# print('Plotting GT Bounding Box')
# # vis.image(np.ones((3, 10, 10)))
# # vis.image(img)
# vis.image(img, opts=dict(caption='Ground Truth Bounding boxes'))

import visdom
import numpy as np
import _init_paths
from datasets.factory import get_imdb
from PIL import Image
import cv2
import torch

# Intializing the visdom server

# vis = visdom.Visdom(server='http://localhost', port='8099')

# Getting the imbd set

imdb = get_imdb('voc_2007_trainval')

# Creating the roidb

roidb = imdb.roidb

imagesIndexList = imdb.image_index

# Creating gt_roidb

gt_roi = imdb.gt_roidb

imagePath_2018 = imdb.image_path_from_index(imagesIndexList[2018])
print(imagePath_2018)

# Loading the image

image2018 = Image.open(imagePath_2018)
image2018Data = np.asarray(image2018)

vis.image(image2018Data.transpose(2, 0, 1))

bounding_boxImages = image2018Data

for i in np.arange(0, 10):
    bounding_boxImages = cv2.rectangle(bounding_boxImages, (roidb[2017]['boxes'][i][0], roidb[2017]['boxes'][i][1]), (roidb[2017]['boxes'][i][2], roidb[2017]['boxes'][i][3]), (255, 0, 0), 2)

vis.image(bounding_boxImages.transpose(2, 0, 1))

roidb[2018]
gt_box = gt_roi()[2018]['boxes']

# Ground truth

image2018 = Image.open(imagePath_2018)
image2018Data = np.asarray(image2018)
gt_bound_image = image2018Data

gt_boundingImage = cv2.rectangle(gt_bound_image, (gt_box[0][0], gt_box[0][1]), (gt_box[0][2], gt_box[0][3]), (255, 0, 0), 2)
vis.image(gt_boundingImage.transpose(2, 0, 1))





