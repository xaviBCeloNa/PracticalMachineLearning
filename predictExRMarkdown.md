# Project Practical Machine Learning
### Predict the manner in which Users did Exercise
==============================================


```
## Loading required package: caret
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

The main objective of the project is to figure out a model that predict how users did exercice (_classe_ variable). You may use any of the other variables to predict with. In order to fullfil the project it is necessary to:
* create a report describing how you built your model, 
* describe how it is used cross validation,
* Explain what you think the expected out of sample error is, and,
* why you made the choices you did

At the end we are going to use our prediction model to predict 20 different test cases.

### 1.- Loading and preprocessing the data

First we load the csv file (both files can be found in [train](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [test](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). The data from the project comes from [project](http://groupware.les.inf.puc-rio.br/har).


```r
myd1.train<-read.csv("pml-training.csv",header=T,na.strings=c("NA",""))
myd1.test<-read.csv("pml-testing.csv",header=T,na.strings=c("NA",""))
dim.train<-dim(myd1.train)
#head(myd)
#str(myd1)
```
Then we can remove variables with no usefull information. We can account for the number of NA's, so any column with a number of NA's long enough will be discarded.


```r
idxToKeep <- c()
for (i in 1:dim.train[2]) {
  if(length(which(!is.na(myd1.train[,i])))==dim.train[1]) { #number of no NA's = to length of rows All data available
    idxToKeep <- c(idxToKeep,i)
  }
}
mydfTrain <- myd1.train[,idxToKeep]
mydfTest <- myd1.test[,idxToKeep]

# Variable x, user_name to num_window will be discarded as well to be non useful 
mydfTrain<-mydfTrain[,8:dim(mydfTrain)[2]]
mydfTest<-mydfTest[,8:dim(mydfTest)[2]]
#names(mydfTrain)
```

It is a good suggestion to clean up near zero variance parameters. In order to deploy this step we are going to use _nearZeroVar_ to our data frame.


```r
nzv <- nearZeroVar(mydfTrain,saveMetrics=TRUE)
# How many variable are ZeroVar
sum(which(nzv$zeroVar==TRUE))
```

```
## [1] 0
```

There is no need to remove further features because all variables are *OK*


## Methods

In order to check preprocessing and cross validation we prepare two sets, data split into 70 and 30 percent.


```r
trainId <- createDataPartition(mydfTrain$classe, list = FALSE, p = 0.7)
trainSet = mydfTrain[trainId, ]
crosvSet = mydfTrain[-trainId, ]

# Make the preprocess model
# We need all columns numeric 
colOk = which(lapply(trainSet, class) %in% c("numeric"))
predMod <- preProcess(trainSet[,colOk], method = c("knnImpute"))

predTrainSet <- cbind(trainSet$classe, predict(predMod,trainSet[,colOk]))
predCrosvSet <- cbind(crosvSet$classe, predict(predMod, crosvSet[, colOk]))

predTestSet <- predict(predMod, mydfTest[, colOk])
names(predTrainSet)[1] <- "classe"
names(predCrosvSet)[1] <- "classe"
```

### Random Forest
Here we build a model based on Random Forest and show the accuracy over training set. Obviously over optimistic (as it is expected). For the cross validation accuracy, the results are pretty reasonable close to 99.5%.


```r
RFMod1 <- randomForest(classe ~ ., predTrainSet)
trainPred <- predict(RFMod1, predTrainSet)
print(confusionMatrix(trainPred, predTrainSet$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
#Cross Validation Accuracy 
crosvSetPred <- predict(RFMod1, predCrosvSet)
print(confusionMatrix(crosvSetPred, predCrosvSet$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668    4    0    0    0
##          B    4 1132    4    0    0
##          C    2    2 1012    3    1
##          D    0    0   10  961    3
##          E    0    1    0    0 1078
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.994    0.986    0.997    0.996
## Specificity             0.999    0.998    0.998    0.997    1.000
## Pos Pred Value          0.998    0.993    0.992    0.987    0.999
## Neg Pred Value          0.999    0.999    0.997    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.283    0.192    0.172    0.163    0.183
## Detection Prevalence    0.284    0.194    0.173    0.166    0.183
## Balanced Accuracy       0.998    0.996    0.992    0.997    0.998
```


## Test 
Here we can check the results of the model over the test data set.



```r
# Our Model
print(RFMod1)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = predTrainSet) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 0.6%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3903    3    0    0    0    0.000768
## B   13 2633   11    1    0    0.009406
## C    0   14 2367   14    1    0.012104
## D    0    0   15 2235    2    0.007549
## E    1    2    1    4 2517    0.003168
```

```r
# Our Estimation
predict(RFMod1, predTestSet)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


## Conclusions
It turns out that the model presented give a really good answer over the _test_ Other models have been tested but discarded to generate poor results (for instance a classification tree with _rpart_) 

The important issue is that before putting our model under the test, we cheched it and did the job quite well. So as a conclusion we never know if our model works fine until it is deployed with a real environment. 

