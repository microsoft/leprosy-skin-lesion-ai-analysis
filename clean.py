# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pandas as pd
import os

'''
Clean poly_df
return a dataframe with columns ['patient_id','image_name','patient_leprosy','type']
'''
def clean_poly(dir_name, # os.path.join(dir_name, 'images') contains all the lesion images
               poly_address, # address of labels.tsv
               patient_info_address, # address of patient information csv
               keep_label_id=False # whether to include the column 'label_id' in the output dataframe
              ):
    poly_df = pd.read_csv(poly_address, sep="\t")
    patient_df=pd.read_csv(patient_info_address, engine='python').replace("200AF", "200AP")
    all_images=os.listdir(os.path.join(dir_name, 'images'))
    
    # These three patients dropped the exp. 
    poly_df=poly_df[~poly_df.patient_id.isin(
        ["Patient-051GR", "Patient-121DF","Patient-203PR"])]
    
    # Drop those rows if corresponding images were not found in the image folder
    poly_df=poly_df[poly_df.image_name.isin(all_images)]
    StudyID=[str[8:] for str in poly_df['patient_id'].values]
    Diag=[patient_df.loc[patient_df['StudyID']==id,'Diagnostic'].values[0] for id in StudyID]
    Diag_patient=['leprosy' if x<2 else 'other_dermotosis' for x in Diag]
    poly_df['patient_leprosy']=Diag_patient
    
    #Remove the lesion if diagnostic result of the patient did not match that of the lesion when the lesion diagnostic result is present. 
    rm_label_id=poly_df.loc[(poly_df['patient_leprosy']!=poly_df['lesion_leprosy']) & poly_df['lesion_leprosy'].notna(),"label_id"].values
    if len(rm_label_id)>0: print("Drop these ids because patient_diag is not equal to lesion_diag: "+', '.join(rm_label_id))
    poly_df=poly_df.loc[~ poly_df['label_id'].isin(rm_label_id),]
    poly_df.drop(columns=['lesion_leprosy'],inplace=True)
    
    # Three image types: closeup, panoramic and edge
    types=[x.split('.')[0].split('-')[-1] for x in poly_df.image_name]
    poly_df['type']=types
    poly_df.replace('paoramic','panoramic',inplace=True)
    poly_df.replace('panoramis','panoramic',inplace=True)
    if keep_label_id: return poly_df[['patient_id','image_name','label_id','patient_leprosy','type']].drop_duplicates()
    else:   
        poly_df=poly_df[['patient_id','image_name','patient_leprosy','type']]
        return poly_df.drop_duplicates()

#return all the image ids of one selected patient
def get_img_ids_onepatient(poly_df,patient_id,type_spec=None):
    df=poly_df
    if type_spec: df=df.loc[df.type==type_spec,]
    selected_label_ids=df.loc[df.patient_id==patient_id,].image_name.values
    return selected_label_ids.tolist()

#return all the image ids of the selected patients
def get_img_ids(poly_df,patient_ids,type_spec=None):
    selected_label_ids=sum([get_img_ids_onepatient(poly_df,p,type_spec) for p in patient_ids],[])
    return selected_label_ids
'''
split a sequence 'seq' into 'num' sub-sequence
return a list including all the sub-sequences [seq1,...,seq_num] 
'''
def chunkIt(seq, num):
    avg = int(len(seq) / num)
    out = []
    last = 0.0
    while len(out)<num:
        if len(out)<num-1: out.append(seq[int(last):int(last + avg)])
        else: out.append(seq[int(last):len(seq)])
        last += avg
    return out


