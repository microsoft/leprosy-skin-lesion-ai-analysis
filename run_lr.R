# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
library(glmnet)
library(pROC)
library(caret)

#' Logistic regression
#' 
#' @param train_df A data frame for model training.
#' @param test_df An optional data frame for model testing. 
#' @param target The column name of the outcome.
#' @param pos_level Will be treated as positive events for classification.
#' @param neg_level Will be treated as negative events for classification.
#' @param nfolds Num of folds for cross validation. Default is 10. 
#' @param foldid An optional vector of values between 1 and nfold identifying what fold each observation is in. If supplied, nfold can be missing.
#' @param alphas Optional user-supplied sequence. Default is [0,.25,.5,.75,1]. Alpha is the elasticnet mixing parameter. When alpha=1, the penalty is the lasso penalty, and alpha=0 the ridge penalty.
#' @param lambdas Optional user-supplied lambda sequence; default is NULL, and glmnet chooses its own sequence.
#' @param show_var Whether to show selected variables. 
#' @param repeated_cv The times to repeat cross-validation to select the optimal alpha. 
#' @return if test_df exisits: list(acc=accuracy,auc=auc,sp=specificity,se=sensitivity) on the testing dataset.
#' @return if test_df=NULL: the trained model (an object of class "cv.glmnet")


run_logit<-function(train_df,
                    test_df=NULL,
                    target='Diagnostic',
                    pos_level='Leprosy',
                    neg_level="OD",
                    nfolds=10,
                    foldid=NULL,
                    alphas=seq(0,1,length.out=5),
                    lambdas=NULL,
                    show_var=FALSE,
                    repeated_cv=1){
  if (any(levels(train_df[[target]])!=c(neg_level,pos_level))) 
    stop("The levels of the target col must match with c(neg_level,pos_level)")
  y<-train_df[,target]
  x <- model.matrix(~., data = train_df[,-which(names(train_df)==target)])
  scores<-c()
  for (ii in 1:repeated_cv){
    # Statified sampling
    if(is.null(foldid)){
      dt<-createFolds(y,k=nfolds)
      foldid<-rep(0,length(y))
      for (ii in 1:length(dt)){foldid[unlist(dt[ii])]<-ii}
    }
    ss<-c()
    for (a in alphas){
      cv=cv.glmnet(x,y,foldid=foldid,alpha=a, family = "binomial",
                   lambda=lambdas,
                   maxit=10000)
      ss<-c(ss,min(cv$cvm))
    }
    scores<-rbind(scores,ss)
  }
  alpha_best<-alphas[which.min(apply(scores,2,mean))]
  logit_model<-cv.glmnet(x,y,foldid=foldid,alpha=alpha_best,
                         family = "binomial",
                         lambda=lambdas,
                         maxit=10000)
  coef_lasso<-coef(logit_model)
  if(show_var){
    print(coef_lasso[rowSums(coef_lasso != 0) != 0,])
  }
  if(!is.null(test_df)){
    test_x=model.matrix(~.,data = test_df[,-which(names(test_df)==target)])
    test_y=test_df[,target]
    pred<-predict(logit_model, test_x,type = "response")
    acc<-mean((pred-.5)*(as.numeric(test_y==pos_level)-0.5)>0)
    pred_output<-as.factor(sapply(pred,function(x){
      if(x>=.5) pos_level
      else neg_level
    }))
    sp<-specificity(pred_output, test_y,negative=neg_level)
    se<-sensitivity(pred_output, test_y,positive = pos_level)
    test_roc = roc(test_y ~ c(pred), plot = FALSE,quiet = TRUE)
    auc=0+test_roc$auc
    list(acc=acc,auc=auc,sp=sp,se=se)
  }else{
    list(model=logit_model)
  }
}

