# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

def derive_transform(size,mean,std,scale=.6,change_aspect_ratio=None):
    augs={
        "size":0,
        "mean":0,
        "std":0,
       "color_contrast": 0.3, 
       "color_saturation": 0.3, 
       "color_brightness": 0.3, 
       "color_hue": 0.1, 
       "rotation": 90, 
       "shear": 20}
    augs['size'] = size
    augs['mean'] = mean
    augs['std'] = std
    tf_list = []
    if not change_aspect_ratio:  tf_list.append(transforms.RandomResizedCrop(augs['size'], scale=(scale, 1),ratio=(1.0, 1.0)))
        #RandomResizedCrop is doing a crop first and then scale to the desired size.
    else: tf_list.append(transforms.RandomResizedCrop(augs['size'], scale=(scale, 1)))
    tf_list.append(transforms.RandomHorizontalFlip())
    tf_list.append(transforms.RandomVerticalFlip())
    tf_list.append(transforms.ColorJitter(
        brightness=augs['color_brightness'],
        contrast=augs['color_contrast'],
        saturation=augs['color_saturation'],
        hue=augs['color_hue']))
    tf_list.append(transforms.ToTensor())
    tf_augment = transforms.Compose(tf_list)
    train_tf=transforms.Compose([tf_augment,transforms.Normalize(augs['mean'], augs['std'])])
    orig_tf=transforms.Compose([transforms.Resize((augs['size'],augs['size'])),transforms.ToTensor(),transforms.Normalize(augs['mean'], augs['std'])])
    return train_tf,orig_tf