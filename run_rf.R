# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
library(randomForest)
library(caret)


#' Random forests
#' 
#' @param train_df A data frame for model training.
#' @param test_df A data frame for model testing. 
#' @param target The column name of the outcome.
#' @param pos_level Will be treated as positive events for classification.
#' @param neg_level Will be treated as negative events for classification.
#' @param show_importance Whether to show feature importance. 
#' @return list(rf_acc=accuracy,rf_auc=auc,sp=specificity,se=sensitivity) on the testing dataset.

rf_pi<-function(train_df,
                test_df,
                target="Diagnostic",
                pos_level='Leprosy',
                neg_level="OD",
                show_importance=FALSE){
  if (any(levels(train_df[[target]])!=c(neg_level,pos_level))) 
    stop("The levels of the target col must match with c(neg_level,pos_level)")
  mtry <- as.integer((ncol(train_df)-1)*c(.6,.8,1))
  ntree<-(1:2)*100
  nodesize=c(1,3)
  sampsize <- as.integer(nrow(train_df)*c(0.6,0.8,1))
  # Create a data frame containing all combinations
  hyper_grid <- expand.grid(mtry = mtry, 
                            nodesize=nodesize, 
                            sampsize = sampsize,
                            ntree=ntree)
  oob_err <- c()
  for (i in 1:nrow(hyper_grid)) {
    model= randomForest(as.formula(paste(paste0("`", target, "`") , " ~ .")),
                        data=train_df, 
                        mtry=hyper_grid$mtry[i],
                        ntree=hyper_grid$ntree[i],
                        nodesize=hyper_grid$nodesize[i],
                        sampsize=hyper_grid$sampsize[i]
                        )
    oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
  }
  opt_i <- which.min(oob_err)
  hyper_grid[opt_i,]
  rf_classifier = randomForest(as.formula(paste(paste0("`", target, "`") , " ~ .")), 
                               data=train_df, 
                               mtry=hyper_grid$mtry[opt_i],
                               ntree=hyper_grid$ntree[opt_i], 
                               nodesize=hyper_grid$nodesize[opt_i],
                               sampsize=hyper_grid$sampsize[opt_i],
                               importance=TRUE)
  if(show_importance) print(create_rfplot(rf_classifier,type=1))
  pred_rf <- predict(rf_classifier,test_df[,-which(names(test_df)==target)])
  prob<-data.frame(predict(rf_classifier,test_df[,-which(names(test_df)==target)],type="prob"))[[pos_level]]
  tab_rf<-table(observed=test_df[,target],predicted=pred_rf)
  sp<-specificity(pred_rf, test_df[,target],negative=neg_level)
  se<-sensitivity(pred_rf, test_df[,target],positive = pos_level)
  acc_rf<-(tab_rf[1,1]+tab_rf[2,2])/sum(tab_rf)
  test_roc = roc(test_df[,target] ~ c(prob), plot = FALSE,quiet = TRUE)
  list(rf_acc= acc_rf,rf_auc=test_roc$auc,sp=sp,se=se)
}


create_rfplot <- function(rf, type=1){
  imp <- importance(rf, type=type, scale = F)
  featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])
  p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
    geom_bar(stat="identity", #fill="#53cfff", 
             #width = 0.65
             ) + coord_flip() + 
    theme_light(base_size=20) +
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.x = element_text(size = 15, color = "black"),
          axis.text.y = element_text(size = 15, color = "black")) 
  return(p)
}

