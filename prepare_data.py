# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels as ptm

class LeprosyDataset(Dataset):
    """Leprosy dataset."""

    def __init__(self, img_names, root_dir, img_df,target_name, transform=None,
            id_col='label_id',
                extension=".jpg"):
        
        """
        Args:
            img_names (string): List of the names of images in this dataset not including extension
            root_dir (string): Image directory.
            img_df (DataFrame): Dataframe including patient_id, image_name, label_id, lesion_leprosy. Its label_id is identitical to the img_names
            target_name (string): the colname of the response variable in img_df
            transform (callable, optional): Transform to be applied on images.
        Returns:
            image: lesion image
            target: 0(leprosy)/1(nonleprosy)
            weight: prop to 1/(num of images of the patient) 
            freq_ratio:  1/(num of images of the patient) 
        """
        self.img_names = img_names
        self.root_dir = root_dir
        self.img_df=img_df
        self.transform = transform
        self.target_name=target_name
        classes = list(self.img_df[self.target_name].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes
        self.extension=extension
        self.id_col=id_col
        lesion_count=[self.img_df.patient_id.values.tolist().count(p) for p in self.img_df.patient_id] # count num of images for each patient
        count_max=max(lesion_count)# max num of images of one patient
        self.count_max=count_max
        self.img_df['sample_weight']=[count_max/x for x in lesion_count] # give an option to re-weight the loss if there are some patients with siginificantly more images than the other. 
        self.img_df['sample_freq_ratio']=[1/x for x in lesion_count]
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.root_dir,
                                self.img_names[i]+self.extension))
        weight=self.img_df.loc[self.img_df[self.id_col]==self.img_names[i],'sample_weight'].values[0]
        freq_ratio=self.img_df.loc[self.img_df[self.id_col]==self.img_names[i],'sample_freq_ratio'].values[0]
        target = self.class_to_idx[self.img_df.loc[self.img_df[self.id_col]==self.img_names[i],self.target_name].values[0]]
        if self.transform:
            image = self.transform(image)
            
        return image, target,weight,freq_ratio
    
    def numpatient(self):
        return len(set(self.img_df.loc[self.img_df[self.id_col].isin(self.img_names),'patient_id'].values))

    
