---
title: "Practical Machine Learning - Project"
author: "James Martin"
date: "September 23, 2015"
output: html_document
---
<!-- rmarkdown v1 -->
## Practical Machine Learning - Project

### Overview

The goal of this project is to build a model predicting how five subjects performed barbell lifts, both correctly and incorrectly. The data was provided by [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har). Both training and testing data was supplied. The models were built using the training data, and the graded submissions were produced by applying them to the testing data.

### Preparing the Data

The training data was loaded from the source file and a subset created further processing. This subset was made up of the direct measurements, omitting identifying variables, timestamps, and summary measures.


```r
# Load the training data
trainingData <- read.csv("C:/Training/Coursera/DataScience/8 - Practical Machine Learning/Project/pml-training.csv")
# Remove extraneous variables
selectedFields <- c(8:10,37:48,60:68,84:86,113:124,151:159,160)
targetData <- trainingData[,selectedFields]
```

The resulting data set was partitioned into a training set and a test set. The training set consists of 70% of the cases, with the remaining 30% held out for cross-validation of the models.


```r
require(caret)
# Selected training cases
inTrain <- createDataPartition(targetData$classe, times=1, p=0.7, list=FALSE)
# Create training data
train <- targetData[inTrain,]
# Create validation data
test <- targetData[-inTrain,]
```

### Building the Models

Since the target variable (**classe**) is a factor with five levels, a tree model seemed to be appropriate. The first attempt was a CART model using all the variables in the reduced training data:


```r
cartFit <- train(classe ~ ., method="rpart", data=train)
```

The results were not terribly impressive, yielding only a 52% accuracy rate on the training data:


```r
cartFit$results
```

```
##           cp  Accuracy      Kappa AccuracySD    KappaSD
## 1 0.03387244 0.5127436 0.36666705 0.02144652 0.03366651
## 2 0.05967518 0.3955300 0.17449326 0.05515520 0.09225659
## 3 0.11392534 0.3287658 0.06816517 0.04106644 0.06171639
```

Cross-validation using the validation data held out from the original sample confirms this poor performance:


```r
cartPred <- predict(cartFit, newdata=test)
cartTab <- table(test$classe, cartPred)
cartTab
```

```
##    cartPred
##        A    B    C    D    E
##   A 1529   28  113    0    4
##   B  482  380  277    0    0
##   C  469   34  523    0    0
##   D  439  184  341    0    0
##   E  158  152  280    0  492
```

```r
cartError <- 1 - ((cartTab[1,1]+cartTab[2,2]+cartTab[3,3]+cartTab[4,4]+cartTab[5,5])/nrow(test))
cartError
```

```
## [1] 0.5031436
```

The CART model predicts an out-of-sample error rate of approximately 50% -- not nearly good enough to submit for grading.

At the risk of overfitting the training data, a random forest model was tried next:


```r
rfFit <- train(classe ~ ., method="rf", data=train)
```

This time, the results were very encouraging:


```r
rfFit$results
```

```
##   mtry  Accuracy     Kappa  AccuracySD     KappaSD
## 1    2 0.9885666 0.9855296 0.001667441 0.002116596
## 2   25 0.9886652 0.9856555 0.002275803 0.002880346
## 3   48 0.9790244 0.9734524 0.004725580 0.005993469
```

A similar cross-validation using the hold-out data supports this:


```r
rfPred <- predict(rfFit, newdata=test)
rfTab <- table(test$classe, rfPred)
rfTab
```

```
##    rfPred
##        A    B    C    D    E
##   A 1672    2    0    0    0
##   B    7 1130    2    0    0
##   C    0    4 1022    0    0
##   D    0    0    5  959    0
##   E    0    0    0    2 1080
```

```r
rfError <- 1 - ((rfTab[1,1]+rfTab[2,2]+rfTab[3,3]+rfTab[4,4]+rfTab[5,5])/nrow(test))
rfError
```

```
## [1] 0.003738318
```

For this model, the expected out-of-sample error rate is approximately 0.4% -- **much** better than the CART model. However, it also seems unrealistically accurate -- overfitting seems to present a definite risk.

### Results

I was delighted to discover that the random forest model correctly predicted all 20 test cases.
