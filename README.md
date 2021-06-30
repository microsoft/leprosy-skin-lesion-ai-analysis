# Introduction 
This project aims to predict the probability of leprosy using skin lesion images and clinical data (as compared to the diagnosis of dermatologists). 
This model is provided for research and development use only. The model is not intended for use in clinical decision-making or for any other clinical use and the performance of model for clinical use has not been established.
# Getting Started
Required R packages in r_requirement.txt
Required python packages in python_requirement.txt

# Build and Test
To replicate cross-validation experiments for Model 1 by resnet50 using close-up images using python:

from run_model import *
for test_id in range(5):
    run_model(
    id=0, # cuda id 
    save_dir=ADDRESS_TO_SAVE_EXP_RESULT,
    experiment_name = 'CV_MODEL1_BY_RESNET50_USING_CLOSEUP',
    batch_size=32,
    tuning='tune_all',
    num_epoch=200,
    model_name='resnet50',
    scale=.6,
    dataset_name='images',
    type_spec='closeup',
    change_aspect_ratio=None,
    num_fold=6,
    test_foldid=test_id) 

Build Model 2 using R

source("clean_metadata.R")
source("run_lr.R")
source("run_rf.R")
source("run_xgboost.R")
dt<-clean_img_metadata(read.csv(ADDRESS_TO_PATIENT_FORM,header=TRUE),
                       read.csv(ADDRESS_TO_LESION_FORM,header=TRUE))
img_metadata<-dt[['img']]
patient_info<-dt[['patient']]
drop<-c("areanearoflesion", "Contralateralarea") # these are highly correlated with the other three temperature features
df<-img_metadata[,-which(names(img_metadata)%in% drop)]
train_df=df[which(df$StudyID %in% PATIENT_TRAIN),] 
test_df=df[which(df$StudyID %in% PATIENT_TEST),] 
df_preprocess<-preprocess(train_df,test_df,cutoff=.8,standardize=TRUE,impute=TRUE)
lr_result<-run_logit(df_preprocess$train,df_preprocess$test,repeated_cv = 5)
xgb_result<-run_xgb(df_preprocess$train,df_preprocess$test,repeated_cv=3)
rf_result<-rf_pi(df_preprocess$train,df_preprocess$test)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


