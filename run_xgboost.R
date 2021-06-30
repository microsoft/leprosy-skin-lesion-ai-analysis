# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
library(ggplot2)
library(dplyr)
library(glmnet)
library(xgboost)
library(mltools)
library(data.table)
library(caret)
#' XGBoosting
#' 
#' @param train_df A data frame for model training.
#' @param test_df A data frame for model testing. 
#' @param nfolds Num of folds for cross validation. Default is 10. 
#' @param target The column name of the outcome.
#' @param repeated_cv The times to repeat cross-validation to select hyperparameters. 
#' @param pos_level Will be treated as positive events for classification.
#' @param neg_level Will be treated as negative events for classification.
#' @return list(acc=accuracy,auc=auc,sp=specificity,se=sensitivity) on the testing dataset.

run_xgb<-function(train_df,
                  test_df,
                  nfolds=10,
                  target="Diagnostic",
                  repeated_cv=1,
                  pos_level='Leprosy',
                  neg_level="OD"
                  ){
  train=train_df
  train_x=(train[,-which(names(train)==target)])
  tempt=one_hot(data.table(train_x))
  train_x=as.matrix(tempt)
  test=test_df
  test_x=(test[,-which(names(test)==target)])
  tempt=one_hot(data.table(test_x))
  test_x=as.matrix(tempt)
  test_y=test[,target]
  control <- trainControl(method = "repeatedcv", repeats = repeated_cv,number = nfolds, 
                          classProbs = TRUE,     
                          summaryFunction = twoClassSummary)
  max_dep<-as.integer(ncol(train_df)/2)
  grid <- expand.grid(nrounds =c(150),
                      eta = c(0.05,0.1,0.3),
                      min_child_weight = 1,
                      colsample_bytree = c(0.6, 0.8),
                      gamma = c(0, 0.25, 0.5),
                      max_depth = c(4,8,12),
                      subsample=.9
  )
  xgb_tune <-train(x= train_x, y= train[,target],
                   method="xgbTree",
                   trControl=control,
                   tuneGrid=grid,
                   metric="ROC"
  )
  pred_output<-predict(xgb_tune, newdata = test_x)
  acc<-mean(pred_output==test[,target])
  prob<-data.frame(predict(xgb_tune, newdata = test_x,type = "prob"))[[pos_level]]
  test_roc = roc(test_y ~ prob, plot = FALSE,quiet = TRUE)
  sp<-specificity(pred_output, test[,target],negative=neg_level)
  se<-sensitivity(pred_output, test[,target],positive = pos_level)
  list(acc=acc,auc=test_roc$auc+0,sp=sp,se=se)
}





