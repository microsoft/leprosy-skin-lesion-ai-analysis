# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy
import os
import cv2
import pandas as pd
import PIL
from PIL import Image
import torchvision.transforms as transforms
import pretrainedmodels as ptm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import numpy as np
from collections import Counter
import random
from clean import *
from preprocess import *
from prepare_data import *
from datetime import datetime

def run_model(
    save_dir, # models and result tables will saved in 'save_dir/experiment_name'. 'save_dir' must exists!
    dir_name, # all images are in dir_name/dataset_name
    poly_address, # address of labels.tsv
    patient_info_address, # address of the patient form csv
    id=0, # cuda id for parallel experiments
    experiment_name = None, # if None, experiment_name will be the date when exp is finished
    lr=.003, # learning rate
    wd=0, # weight decay
    batch_size=32,
    tuning='tune_all', # default is to train the whole neural net. Otherwise train the last layer only. 
    class_weight="no", # not apply class_weight to the loss. Otherwise do apply. 
    sample_weight="no",# not apply sample_weight to the loss. Otherwise do apply. 
    num_epoch=200, # num of epochs
    model_name='resnet50',# the other option 'inceptionv4'
    no_augmentation=False, # if TRUE -> no augmentation at all
    change_aspect_ratio=None, # whether to change aspect ratio
    scale=.6, # scaling factor for data augmentation
    dataset_name='images', # all images are in dir_name/dataset_name
    type_spec=None,# If None, use all three types of images. If ='closeup'/'panoramic'/'edge', only use the specific type of images for model training/validation/testing.
    optim_type="SGD", # otherwise use "Adam" as the optimizer
    num_fold=6, # Split all patients into 6 folds. The last folds will be used to validate Model 3 later. The first 5 folds will be used for 5-fold cross-validation. 
    test_foldid=0 # index of the validation fold. 0-4 only.
):
    if type_spec not in [None,'closeup','panoramic','edge']: raise IOError('Invalid image type!')
    if not os.path.exists(save_dir): raise IOError('save_dir not exisit!')
    device=torch.device("cuda",id)
    poly_df=clean_poly(dir_name,
                   poly_address,
                   patient_info_address)
    patient_list=poly_df.patient_id.unique()
    patient_list_leprosy=poly_df.loc[poly_df.patient_leprosy=='leprosy','patient_id'].unique()
    patient_list_OD=poly_df.loc[poly_df.patient_leprosy=='other_dermotosis','patient_id'].unique()
    leprosy_split=chunkIt(patient_list_leprosy, num_fold)
    OD_split=chunkIt(patient_list_OD, num_fold)

    # Levae the last folder out for validating Model 3
    leprosy_split.pop()
    OD_split.pop()
    # Fold 'test_foldid' for testing, one for model selection by the best accuracy, and the rest for model training. This is a stratified split. 
    test_patient=np.concatenate((leprosy_split.pop(test_foldid), 
                             OD_split.pop(test_foldid) ), axis=None)
    val_patient=np.concatenate((leprosy_split.pop(), 
                             OD_split.pop()), axis=None)
    train_patient=np.concatenate(leprosy_split+OD_split)
    
    # Drop the three patients without any closeup image
    poly_df=poly_df[~poly_df.patient_id.isin(
        ['Patient-009OS', 'Patient-017SG', 'Patient-076PR'])]
    
    # Get all the image ids of the test/val/train patients
    test_img_ids=get_img_ids(poly_df,test_patient,type_spec)
    train_img_ids=get_img_ids(poly_df,train_patient,type_spec)
    val_img_ids=get_img_ids(poly_df,val_patient,type_spec)
    poly_df=poly_df.loc[poly_df.image_name.isin(train_img_ids+val_img_ids+test_img_ids)]
      
    train_diag=poly_df.loc[poly_df.image_name.isin(train_img_ids),'patient_leprosy'].values
    max_cnt=max(train_diag.tolist().count('leprosy'),train_diag.tolist().count('other_dermotosis'))
    if class_weight!="no": class_weights=torch.FloatTensor([max_cnt/train_diag.tolist().count('leprosy'),
      max_cnt/train_diag.tolist().count('other_dermotosis')]).to(device)

    if model_name=='resnet50': model=ptm.resnet50(num_classes=1000, pretrained='imagenet')
    else: model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
    model.last_linear = nn.Linear(model.last_linear.in_features, 2)
    if torch.cuda.is_available():model.cuda(id)
    train_tf,test_tf=derive_transform(model.input_size[1],model.mean,model.std,scale,change_aspect_ratio)
    val_tf=test_tf
    if no_augmentation: train_tf=test_tf
    img_address=os.path.join(dir_name, dataset_name)
    
    # Includes only one type of images if specified
    if type_spec: poly_df=poly_df.loc[poly_df.type==type_spec,]
    train_data=LeprosyDataset(train_img_ids, img_address, poly_df,target_name='patient_leprosy',transform=train_tf,
                         id_col='image_name',extension="")
    train_patient_num=train_data.numpatient()
    train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    val_data=LeprosyDataset(val_img_ids, img_address, poly_df,'patient_leprosy',val_tf,
                       id_col='image_name',extension="")
    val_patient_num=val_data.numpatient()
    val_loader=DataLoader(val_data,batch_size=batch_size,num_workers=4)
    test_data=LeprosyDataset(test_img_ids, img_address, poly_df,'patient_leprosy',test_tf,
                        id_col='image_name',extension="")
    test_patient_num=test_data.numpatient()
    test_loader=DataLoader(test_data,batch_size=batch_size,num_workers=4)
    if class_weight=="no":
        criterion = nn.CrossEntropyLoss(reduction='none')
    else: criterion = nn.CrossEntropyLoss(reduction='none',weight=class_weights)
    if(optim_type=="SGD"):
        if tuning=='tune_all':
            optimizer = optim.SGD(model.parameters(),lr=lr,weight_decay=wd)
        else: optimizer = optim.SGD(model.last_linear.parameters(),lr=lr,weight_decay=wd)
    else:
        if tuning=='tune_all': optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
        else: optimizer = optim.Adam(model.last_linear.parameters(),lr=lr,weight_decay=wd)       
    train_loss,val_loss,test_loss=[],[],[]
    best_val=1000.0
    best_val_acc=0.0
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        
        # model training
        model.train()
        for i,data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels,weights,ratios = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights=weights.float()
            weights=weights.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if sample_weight!="no": loss = loss * weights
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.sum()
        confusion_matrix = torch.zeros(2,2)
        correct,total,correct_p = 0,0,0
        running_loss = 0.0
        with torch.no_grad():
            for data in train_loader:
                images, labels,_,ratios = data
                images=images.to(device)
                labels=labels.to(device)
                ratios=ratios.float()
                ratios=ratios.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.sum()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_p += ((predicted == labels).float()*ratios).sum().item() #patient_wise average
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        print('[%d] loss: %.3f' %(epoch + 1, running_loss))
        print('Accuracy of the network on the training images: %d %%' % (100 * correct / total))
        print(correct_p/train_patient_num)
        train_loss.append([running_loss.item(),100 * correct / total,100*correct_p/train_patient_num]+
                      confusion_matrix.reshape(1,-1)[0].tolist())
        
        # model validation 
        confusion_matrix = torch.zeros(2,2)
        correct,total,correct_p = 0,0,0
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                images, labels,_ ,ratios= data
                images=images.to(device)
                labels=labels.to(device)
                ratios=ratios.float()
                ratios=ratios.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.sum()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_p += ((predicted == labels).float()*ratios).sum().item() 
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        print('Accuracy of the network on the val images: %d %%' % (100 * correct / total))
        print(correct_p/val_patient_num)
        val_loss.append([running_loss.item(),100 * correct / total,100*correct_p/val_patient_num]+
                        confusion_matrix.reshape(1,-1)[0].tolist())
        
        # if val_loss<best_val, update best_model
        if running_loss/total<best_val:
            best_model=copy.deepcopy(model).state_dict()
            best_val=running_loss/total
            
        # if val acc>best_val_acc, update best_model_acc
        if correct/total>best_val_acc:
            best_model_acc=copy.deepcopy(model).state_dict()
            best_val_acc=correct/total
        
        # model testing
        confusion_matrix = torch.zeros(2,2)
        correct,total,correct_p = 0,0,0
        running_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels,_,ratios = data
                images=images.to(device)
                labels=labels.to(device)
                ratios=ratios.float()
                ratios=ratios.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.sum()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_p += ((predicted == labels).float()*ratios).sum().item() 
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        print(correct_p/test_patient_num)
        test_loss.append([running_loss.item(),100 * correct / total,100*correct_p/test_patient_num]+
                         confusion_matrix.reshape(1,-1)[0].tolist())
    print('Finished Training')
    # Save outputs
    if not experiment_name:
        execution_time = datetime.now().strftime('%d-%m-%Y-%H-%M')
        experiment_name = execution_time
    experiment_folder = os.path.join(save_dir,experiment_name,'fold'+str(test_foldid))
    if not os.path.exists(experiment_folder): os.makedirs(experiment_folder)
    torch.save(model.state_dict(), os.path.join(experiment_folder,'model.pt'))
    torch.save(best_model, os.path.join(experiment_folder,"best_val__model.pt"))
    torch.save(best_model_acc, os.path.join(experiment_folder,"best_val_acc_model.pt"))
    pd.DataFrame(train_loss).to_csv(os.path.join(experiment_folder,'train.csv'),index=False)
    pd.DataFrame(val_loss).to_csv(os.path.join(experiment_folder,'val.csv'),index=False)
    pd.DataFrame(test_loss).to_csv(os.path.join(experiment_folder,'test.csv'),index=False)
    return None

