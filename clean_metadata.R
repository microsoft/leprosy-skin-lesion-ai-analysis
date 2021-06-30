# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
library(dplyr)
library(summarytools)
library(ggplot2)
library(ggpubr)
library(corrplot)
library(caret)

#' Impute a specific column
#' For each patient, a missing value will be imputed by the median of values for the other lesions from the same patient. If all values are missing for this column, impute with the median of the whole dataset. 
#' @param dt A data frame.
#' @param col_name The name of the column to be imputed.  
#' @param global_med An optional global value to impute missing values with. If missing, will be the median of the column.  
#' @param id_col_name The column name of patient ids. 
#' @return list(impute=imputed column,impute_val=global_med)

impute_column<-function(dt,col_name,global_med=NULL,id_col_name='StudyID'){
  if(!id_col_name %in% colnames(dt)) stop(paste("The column",id_col_name,"does not exist!",sep=" "))
  if(is.null(global_med))global_med<-median(dt[,col_name],na.rm=TRUE)
  impute<-as.numeric(apply(dt,1,function(row){
    x=row[col_name]
    if(is.na(x)){
      id=row[id_col_name]
      me<-median((dt[,col_name])[which(dt[[id_col_name]]==id)],na.rm = TRUE)
      if(is.na(me)) global_med else me
    }else x 
  }))
  list(impute=impute,impute_val=global_med)
}
#' Convert a string to a number to pre-process temperatures.
#'
#' @param x A string 'xx,x'
#' @return x A number 'xx.x'
#' 
#' @example transfer_temp('30,1')
#' @example transfer_temp('30')
transfer_temp<-function(x){
  if(!is.na(x)){
    if(grepl(',',x)){
      dec<-unlist(strsplit(as.character(x),","))
      as.numeric(dec[1])+as.numeric(dec[2])/10 
    }else as.numeric(as.character(x))
  }else x
}

#' Data cleaning
#' 
#' @param patient_info A data frame containing patient information.
#' @param img_metadata A data frame containing metadata. 
#' @param target The column name of the outcome.
#' @param pos_level Will be treated as positive events for classification.
#' @param neg_level Will be treated as negative events for classification.
#' @param keep_Lesionregister Whether to include the column 'Lesionregister' in the returned dataframe img.
#' @return list(patient=patient_info,img=img_metadata).

clean_img_metadata<-function(patient_info,img_metadata,
                             target='Diagnostic',
                             pos_level='Leprosy',
                             neg_level="OD",
                             keep_Lesionregister=FALSE){
  ######################
  # Clean patient info
  ######################
  diagnositic_tests=c("RidleyJopling","Baciloscopy","IB","PCR","Histopathologicalanalyses","AntiPGL1")
  rm=c("Clinicalformofindexcase","Whichnerves","Whichdermatose","StudyID.1","UniqueKey")
  patient_info$Diagnostic<-sapply(patient_info$Diagnostic,function(x){
    if(is.na(x)){ x
    }else if(x==2){
      "OD"
    }else "Leprosy" 
  })
  # Rm empty and unnecessary cols
  img_metadata=(img_metadata[,colSums(is.na(img_metadata))!=nrow(img_metadata)])
  df=(patient_info[, -which(names(patient_info) %in% c(diagnositic_tests,rm))])
  df=(df[,colSums(is.na(df))!=nrow(df)])
  df$Diagnostic<-as.factor(df$Diagnostic)
  
  df$Age<-sapply(df$BirthDate,function(x){
    str<-as.character(x)
    year<-as.integer(unlist(strsplit(str,"/"))[3])
    2019-year
  })
  df<-df[,-which(names(df)=="BirthDate")]
  df$HouseHoldContact[is.na(df$HouseHoldContact)]<-0
  df$HouseHoldContact<-as.factor(df$HouseHoldContact)
  df$DurationofLesion[is.na(df$DurationofLesion)]<-0
  df[,'DurationofLesion']=as.factor(df[,'DurationofLesion'])
  df$NumberofLesions[is.na(df$NumberofLesions)]<-0
  df[,"NumberofLesions"]=as.factor(df[,"NumberofLesions"])
  df=droplevels(df)
  # Group Nodules together with Papules
  df[,'NodulesPapules']=as.factor(mapply(function(a,b){
    a=="True" | b=="True"
  }, df$Papules,df$Nodules))
  df=select(df,-'Papules',-'Nodules')
  # Drop near-zero variance variables
  nearZeroIndex<-nearZeroVar(df,freqCut=nrow(df)/9,uniqueCut = 5)
  if(length(nearZeroIndex)>0) df<-df[,-nearZeroIndex]
  patient_info<-df
  
  ######################
  # Clean metadata
  ######################
  # All "" indicates NA
  img_metadata[img_metadata==""]<-NA
  img_metadata$Diagnostic<-sapply(img_metadata$StudyID,function(id){
    d<-patient_info[patient_info$StudyID==as.character(id),"Diagnostic"]
    if (length(d)==0) NA else as.character(d)
  })
  
  #Drop all rows with no diagnostic result
  img_metadata<-img_metadata[!is.na(img_metadata$Diagnostic),]
  img_metadata$Diagnostic<-as.factor(img_metadata$Diagnostic)
  drop<-c("PanoramicPhoto", "CloseupPhoto", "Edgeofthelesion","Lesionregister","NotaplicablpeTemp")
  if(keep_Lesionregister) Lesionregister<-img_metadata$Lesionregister
  img_metadata<-img_metadata[,-which(names(img_metadata)%in% drop)]
  
  # Group papule with nodule
  img_metadata$type[img_metadata$type %in% c(1,3)]<-13
  img_metadata$type<-as.factor(img_metadata$type)
  
  #Rename colnames
  colnames(img_metadata)[which(colnames(img_metadata)=="Color")]<-"color"
  img_metadata$color<-as.factor(img_metadata$color)
  
  # If Sensoryloss is missing, treat it as the third category, since it indicates a small lesion. 
  img_metadata$Sensoryloss[is.na(img_metadata$Sensoryloss)]<-2
  img_metadata$Sensoryloss<-as.factor(img_metadata$Sensoryloss)
  
  img_metadata$Site<-as.factor(img_metadata$Site)

  img_metadata$Diameter<-sapply(img_metadata$Diameter,function(x){
    gsub(",", ".", x)
  })
  img_metadata$Diameter<-as.numeric(as.character(img_metadata$Diameter))
  
  img_metadata$atlesion<-sapply(img_metadata$atlesion,transfer_temp)
  img_metadata$areanearoflesion<-sapply(img_metadata$areanearoflesion,transfer_temp)
  img_metadata$Contralateralarea<-sapply(img_metadata$Contralateralarea,transfer_temp)
  img_metadata$diff_at_contra<-img_metadata$atlesion-img_metadata$Contralateralarea
  img_metadata$diff_at_near<-img_metadata$atlesion-img_metadata$areanearoflesion

  img_metadata<-droplevels(img_metadata)
  nearZeroIndex<-nearZeroVar(img_metadata,freqCut=nrow(img_metadata)/9,uniqueCut = 5)
  if(length(nearZeroIndex)>0)img_metadata<-img_metadata[,-nearZeroIndex]
  
  patient_info<-df
  img_metadata[,target]<-factor(img_metadata[,target],
                                levels=c(neg_level,pos_level))
  patient_info[,target]<-factor(patient_info[,target],
                                levels=c(neg_level,pos_level))
  if(keep_Lesionregister) {
    print(length(Lesionregister))
    print(nrow(img_metadata))
    img_metadata$Lesionregister<-Lesionregister}
  list(patient=patient_info,img=img_metadata)
}
#' Data pre-processing for model training and testing.
#' 
#' @param train_df The training data frame.
#' @param test_df An optional testing data frame. 
#' @param cutoff A numeric value for the pair-wise absolute correlation cutoff. Variables will be removed untill there is no highly correlated pairs. 
#' @param standardize Whether to standardize numerical cols. 
#' @param impute Whether to impute missing values. 
#' @return list(patient=patient_info,img=img_metadata,mean=mean for standardization,std=standard deviation for standardization).
preprocess<-function(train_df,test_df=NULL,cutoff=.8,standardize=TRUE,impute=TRUE){
  mean_s<-c()
  std_s<-c()
  if(impute){
    train_df$Diameter[is.na(train_df$Diameter)]<-1
    ss<-impute_column(train_df,"atlesion")
    impute_atlesion<-ss$impute_val
    train_df$atlesion<-ss$impute
    ss<-impute_column(train_df,"diff_at_contra")
    impute_diff_at_contra<-ss$impute_val
    train_df$diff_at_contra<-ss$impute
    ss<-impute_column(train_df,"diff_at_near")
    impute_diff_at_near<-ss$impute_val
    train_df$diff_at_near<-ss$impute
    if(!is.null(test_df)){
      test_df$Diameter[is.na(test_df$Diameter)]<-1
      test_df$atlesion<-impute_column(test_df,"atlesion",impute_atlesion)$impute
      test_df$diff_at_contra<-impute_column(test_df,"diff_at_contra",impute_diff_at_contra)$impute
      test_df$diff_at_near<-impute_column(test_df,"diff_at_near",impute_diff_at_near)$impute
      }
  }
  if ('StudyID'%in%names(train_df)) train_df=select(train_df,-'StudyID')
  if ('StudyID'%in%names(test_df)) test_df=select(test_df,-'StudyID')
  for (ii in 1:ncol(train_df)){
      if (is.numeric(train_df[,ii])){
        if(standardize){
          m=mean(train_df[,ii],na.rm=T)
          std=sd(train_df[,ii],na.rm=T)
          mean_s=c(mean_s,m)
          std_s=c(std_s,std)
          train_df[,ii]=(train_df[,ii]-m)/std
          if(!is.null(test_df)) test_df[,ii]=(test_df[,ii]-m)/std
        }
      }

  }
  if(sum(sapply(train_df,is.numeric))>1){
    cor_df<-cor(na.omit(train_df[,sapply(train_df,is.numeric)]))
    rm_col<-names(train_df[,sapply(train_df,is.numeric)])[findCorrelation(cor_df,cutoff=cutoff)]
    if(length(rm_col)>0){
      train_df=select(train_df,-all_of(rm_col))
      if(!is.null(test_df)) test_df=select(test_df,-all_of(rm_col))
    }
  }
  list(train=train_df,test=test_df,mean=mean_s,std=std_s)
}

