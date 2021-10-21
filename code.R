#### Code accompanying Facial recognition technology can expose political orientation from naturalistic facial images
#### https://doi.org/10.1038/s41598-020-79310-1
##
## To preserve participants' anonymity, a small amount of random noise has been added to all continuous variables.
## To prevent database-wide matching, a tiny percentage of values in categorical variables (e.g., gender, ethnicity) has been randomly changed.     
## Thus, the results may be minimally different from those quoted in the paper.  
##
## The code has been simplified for clarity. 
## I would recommend using parallel processing to compute the results. (It has been removed for compatibility with different R implementations.)

#########################################################
### Load libraries
#########################################################
library(glmnet)
library(pROC)
library(data.table)

#########################################################
### Load data 
#########################################################
load("faces.RData")
colnames(d)

#### Variables
# userid: anonymous user id
# gender: self-reported, 1=female / 0=male
# age: self-reported, in 2019
# country: self-reported
# facial_hair: probability, see manuscript
## Political views (0 = liberal / 1= conservative)
# pol: all samples
# pol_dat_[us/ca/uk]: dating sample, US/CA/UK
# pol_fb_us: Facebook sample, US                              
## Big 5Personality (standardized, mean=0, SD=1)
# ext: Extroversion 
# neu: Neuroticism
# ope: Openness
# agr: Agreeableness
# con: Conscientiousness 
# database: dating / fb 
## Following variables were estimated usingFacePlusPlus.com API (FPP)
# emotion.[sadness/neutral/disgust/anger/surprise/fear/happiness]: Emotional expression 
# gender.value: 1=female / 0=male
# age.value
# headpose.[yaw_angle/pitch_angle/roll_angle                    
# smile.value
# [left/right]_eye_status.[normal_glass_eye_open/no_glass_eye_close/occlusion/no_glass_eye_open/normal_glass_eye_close/dark_glasses]
# ethnicity.value"                        

#########################################################
######################## TABLE 1 ########################
#########################################################
table(d$country, d$pol, d$database)
by(d$gender, list(d$pol, d$country, d$database), function(x) round(mean(x, na.rm=T),3))
by(d$ethnicity.value, list(d$pol, d$country, d$database), function(x) round(mean(x=="white", na.rm=T),3))
by(d$age, list(d$pol, d$country, d$database), function(x) round(summary(x, na.rm=T),3))
### Totals
mean(d$gender,na.rm=T)
round(mean(d$ethnicity.value=="white", na.rm=T),2)
summary(d$age)
#########################################################
###################### end of TABLE 1 ###################
#########################################################

#########################################################
#################### Train the models ###################
#########################################################
ground_truth<-c("pol_dat_us", "pol_dat_ca","pol_dat_uk","pol_fb_us")

## Set the number of cross validation folds
cv_folds<-4 #cv_folds=30 is used in the paper
fold<-sample(1:cv_folds, nrow(d),T)

#### Function computing Area Under the ROC Curve
AUC<-function(predictions, labels){
  f <- which(!is.na(labels) & !is.na(predictions))
  pred <- roc(labels[f], predictions[f], quiet = T)
  ci <- ci.auc(pred)
  out <- c(as.numeric(pred$auc), as.numeric(ci)[-2])
  names(out) <- c("auc", "l_ci", "u_ci")
  return(round(out, 4))
}

## Load face descriptors
load("./vgg.RData")

for (i in ground_truth){
  print(i)
  var<-paste("pred_",i,sep="")
  d[[var]]<-NA
  f<-which(!is.na(d[[i]]))
  
  for (j in 1:cv_folds){;cat(".")
    train<-f[fold[f]!=j]
    test<-f[fold[f]==j]
    
    ####### Estimate Lambda and fit the model on the training data  
    # Consider using parallel processing (parallel=T)
    fit <- cv.glmnet(y=d[[i]][train],x=vgg[train,],nfolds = cv_folds, alpha=1,family="binomial",standardize = T,intercept = T)
    ####### Estimate the values for the test data  
    d[[var]][test]<-predict.glmnet(object=fit$glmnet.fit,newx=vgg[test,], type="response", s=fit$lambda.min)

    ####### Estimate the values for faces in other samples (for Table 2)
    # Consider using parallel processing (e.g., the foreach loop)
    fx<-which(is.na(d[[var]]) & fold==j)
    d[[var]][fx]<-predict.glmnet(object=fit$glmnet.fit,newx=vgg[fx,], type="response", s=fit$lambda.min)    
  }
}    

#########################################################
##### Figure 2 and Table 2 ##############################
##### Estimate AUC and Confidence Intervals #############
##### All models by all models ##########################
#########################################################
predicted_values<-paste("pred_", ground_truth, sep="")
AUC_all<-matrix(nrow = length(predicted_values),ncol = length(ground_truth),dimnames = list(predicted_values,ground_truth))
CI_all<-matrix(nrow = length(predicted_values),ncol = length(ground_truth),dimnames = list(predicted_values,ground_truth))

for (i in predicted_values){
  for (j in ground_truth){;cat(".")
    x<-AUC(d[[i]], d[[j]])
    AUC_all[i,j]<-x[1]
    CI_all[i,j]<-x[3]-x[1]
  }
}
print(round(AUC_all,3))
print(round(CI_all,3))

#########################################################
#### Results presented in Figure 2 ######################
#### AUC when controlling for gender, age, ethnicity ####
#########################################################

#### Function for creating user pairs
create_user_grid<- function(uid,age,gender,pol, max_age_diff){
  d1<-data.frame(uid,age=round(age),gender,pol=factor(pol))
  f<-which(!duplicated(uid) & rowSums(is.na(d1))==0)
  d1<-d1[f,];d1$id<-1:nrow(d1)
  d2<-split(d1$id, list(d1$pol,d1$gender, d1$age))
  out<-list()
  for (au in sort(unique(age))){
    for (gu in 0:1){
      lib<-d2[[paste("lib",gu,au,sep=".")]]
      con<-as.numeric(unlist(d2[names(d2) %in% paste("con",gu,ceiling(au-(max_age_diff/2)):floor(au+(max_age_diff/2)),sep=".")]))
      if (min(c(length(lib), length(con)))==0) next()
      out[[paste("lib",gu,au,sep=".")]]<-expand.grid(lib,con);#cat(".")
    }
  }
  out<-rbindlist(out);rownames(out)<-NULL
  if (ncol(out)==2) colnames(out)<-c("lib", "con")
  out$gender<-d1$gender[out$lib]
  out$lib<-d1$uid[out$lib];out$con<-d1$uid[out$con]
  return(out)
}

### compute the results
for (j in ground_truth){
  f<-!is.na(d[[j]])
  tmp<-d[f,]
  tmp[[j]]<-c("con","lib")[tmp[[j]]+1]
  tmp<-split(tmp, list(g=tmp$gender,  e=tmp$ethnicity.value),drop = T)
  tmp<-tmp[sapply(tmp, function(x) length(table(x[[j]]))>1)]
  tmp<-tmp[sapply(tmp, nrow)>2]
  
  for (i in names(tmp)) tmp[[i]]<-create_user_grid(uid=tmp[[i]]$userid,age=round(tmp[[i]]$age.value),
                                                   gender=tmp[[i]]$gender,pol=tmp[[i]][[j]], max_age_diff = 1)
  tmp<-do.call(rbind, tmp)
  tmp$liberal<-d[[paste("pred_", j, sep="")]][match(tmp$lib,d$userid, incomparables = NA)]
  tmp$conservative<-d[[paste("pred_", j, sep="")]][match(tmp$con,d$userid, incomparables = NA)]
  print(paste(j, ": ", round(mean(tmp$liberal>tmp$conservative),2),sep=""))
}

###### NOTE:
# Those results seem to be somewhat affected by the noise
# added to gender/age/ethnicity. The accuracies are 
# about .02 below those reported on Figure 2


#########################################################
### Results presented in Table S1 and Figure 3 ##########
### Predictive power offered by interpretable features ##
#########################################################
# The code below can be modified to compute accuracies for
# for any combination of variables and any database

cv_folds<-4
fold<-sample(1:cv_folds, nrow(d),T)
db<-"fb"

# prepare head orientation for the analysis
d$headpose1.pitch_angle<-abs(d$headpose.pitch_angle)
d$headpose1.yaw_angle<-abs(d$headpose.yaw_angle)
d$headpose1.roll_angle<-abs(d$headpose.roll_angle)

# list of variables
vars<-list(headpose=c("headpose1.pitch_angle","headpose1.yaw_angle","headpose1.roll_angle"),
           expressions=c("emotion.sadness","emotion.neutral","emotion.disgust","emotion.anger","emotion.surprise","emotion.fear","emotion.happiness"), 
           personality =c("ope", "con", "ext", "agr", "neu"))

# train the models
for (v in vars){
  print(v)
  var<-paste("pred_",paste(v, collapse="_"),sep="")
  d[[var]]<-NA
  f<-which(d$database==db & !is.na(rowSums(d[,v, drop=F])))
  
  for (j in 1:cv_folds){;cat(".")
    train<-f[fold[f]!=j]
    test<-f[fold[f]==j]
    ####### Estimate Lambda and fit the model on the training data  
    fit <- cv.glmnet(y=d$pol[train],x=as.matrix(d[train,v]),nfolds = cv_folds, alpha=1,family="binomial",standardize = T,intercept = T)
    ####### Estimate the values for the test data  
    d[[var]][test]<-predict.glmnet(object=fit$glmnet.fit,newx=as.matrix(d[test,v]), type="response", s=fit$lambda.min)
  }
  #### print AUC
  print(AUC(d[[var]], d$pol))
}
