## Practical Machine Leaning - Course Project Writeup

This project (Weight Lifting Exercise analysis) is about building model to determine how well people perform barbell lifts with data from their wearing sensors; 
Training data came from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, which is collected from (this source)[http://groupware.les.inf.puc-rio.br/har]. The accuracy of my model -- built by using PCA with threshold 95% and Random Forest -- is about 96.8% on training dataset and 98 % for validation set, and 95% on 19/20 testing set.

### Setting up the required libraries and getting the required data
```
library(caret)
library(VIM)
library(gridExtra)
library(knitr)
library(doMC)
registerDoMC(cores=2) # For parallel random forest
opts_chunk$set(cache=TRUE,echo=TRUE)
```
If above libraries are not present already, then install then using command 'intall.packages(<libraryname>)'
Make sure that you have downloaded the following files from below links for running this script (must put these files in same folder as this script) 
Training set - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
Testing set - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### A look at the dataset
```
names(orgData)
```
Many are statistical purpose variables which give something like summary of given sensors data within a time window. 
It maybe used to fix the noise of data given by sensors, but to simplify, now just removing it from analysis.
```
ignored_column_regx <- "^(kurtosis|skewness|min|max|stddev|total|var|avg|ampl)"
data <- orgData[,grep(ignored_column_regx,names(orgData),invert=T)]
```
Next, checking for incomplete data values
```
table(complete.cases(data))
```
No incomplete data was found, so continuing without dealing with imputing. If we go backward and plugging in statistical variables above, 
we can see that almost all cases (nearly 97%) are incomplete leading to some serious working in imputing.
```
a <- aggr(orgData,plot=F)
#summary(a)
table(a$missings$Count)
```
Some variables can also be removed:

* New window yes mean it ist time to calculate statistical variables, we remove all stat vars
* user_name contains user name, our output model should be applied to any other user regardless of their name; So ignoring this variable too
* cvtd_timestamp is timestamp but in another form so it can be removed
* X and num_window show the order, I assume that sensor device works stable regardless of how long it has run. 
We notice that timestamp should be kept because at night the activity of body may go down and it related to the how they do. 

### Create training dataset and validation dataset
```
data <- data[,grep("^(num_window|cvtd_timestamp|X|new_window)",names(data),invert=T)]
inTraining <- createDataPartition(y=data$classe,p=.75,list=F) 
training <- data[inTraining,]
validation <- data[-inTraining,]

train_predictors <- training[,-c(52)]
train_outcome <- training[,c(52)]

validation_pred <- validation[,-c(52)]
validation_outcome <- validation[,c(52)]
```

### Now exploring how classes are seperated by 2d plot
```
train_predictors_scaled <- scale(train_predictors[,c(-1)],center=T,scale=T)
pc <- prcomp(train_predictors_scaled) 
cumsum((pc$sdev^2 / sum(pc$sdev^2))) # First 2 exlain only (30% variance), but just exlore to see any pattern heres ?
training_predictors_pc12 <- as.matrix(train_predictors_scaled) %*% pc$rotation[,c(1,2)]
training_predictors_pc12 <- data.frame(x=training_predictors_pc12[,1],y=training_predictors_pc12[,2])
q1<-qplot(data=training_predictors_pc12,x=x,y=y,col=train_outcome)
q2<-qplot(data=training_predictors_pc12,x=x,y=y,col=train_predictors[,c(1)])
grid.arrange(q1,q2)
```
It seem that first 2 layer of PCA only seperate 6 people quite well , but not sure how to separate how they do.

Use PCA to reduce complexity before training
```
train_predictors <- training[,-c(1,52)]
train_outcome <- training[,c(52)]
```
PCA with Centering and Scaleing togother because of timestamp are very big value and I will use
```
preProcess <- preProcess(train_predictors,method=c("center","scale","pca"),thresh=.95)
# Preprocess training and validation set
train_pred_preprocessed <- predict(preProcess,train_predictors)
validation_pred_preprocessed <- predict(preProcess,validation_pred[,-c(1)])
```
Cross-validation with 3 fold, and allow running for paralell for shorten time
```
myCtrl <- trainControl(method="cv",number=3,allowParallel=T)
```

Now use tree to classify
```
sepTree <- train(train_outcome~.,method="rpart",data=train_pred_preprocessed,trControl=myCtrl)
sepTree$results 
```

Only 39% accuracy, so need more powerful model. Trying lda:
```
sepLda <- train(train_outcome~.,method="lda",data=train_pred_preprocessed,trControl=myCtrl)
sepLda$results
```
Nearly 52%, a little bit better but not enough. Try more powerful model, random forest
```
sepRF <- train(train_outcome~.,data=train_pred_preprocessed,method="rf",trControl=myCtrl,allowParalell=T)
sepRF
confusionMatrix(predict(sepRF,validation_pred_preprocessed),validation_outcome)
```
Random forest performance on training set is 96.8% and on validation is 98%. 
This mean it maybe quite good on out-of-sample error ( based on our validation set performance). No need to justify model more.

### Calculating on test set

Now testing on test set
```
test_predictors <- orgTest[,names(train_predictors)]
test_predictors_preprocessed <- predict(preProcess,test_predictors)
predict(sepRF,test_predictors_preprocessed)
# 19/20 right, 1 wrong classify at position 3
```

### Reference
* http://groupware.les.inf.puc-rio.br/har
