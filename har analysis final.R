# Coursera: Practical Machine Learning
# Project 1: Human Activity Recognition
# Final script
# Author: Wai-Wah Liu

################## Pre processsing #######################
# Set working directory
setwd("H:\\Coursera\\Data Science Track\\08_Practical Machine Learning\\week 3\\Project 1")

# Training data
train <- read.csv("H:\\Coursera\\Data Science Track\\08_Practical Machine Learning\\week 3\\Project 1\\pml-training.csv",
 header=TRUE, stringsAsFactors=FALSE, sep=",", na.strings="NA")

nrow(train) # 19622

# NA per numeric variable
NAperVar <- NA

for (j in 1:ncol(train))
 { 
  NAperVar[j] <- sum(is.na(train[ ,j])) 
 }

# NA per "string" variable
NAperVar2 <- NA

for (j in 1:ncol(train))
 {
  NAperVar2[j] <- sum(train[ ,j]=="")  
 }

print(NAperVar)
print(NAperVar2)

# Cases met > 19000 NA's verwijderen (allemaal 19216 NA's)
remove <- c(which(NAperVar > 19000), which(NAperVar2 > 19000))

train <- train[ ,-remove] # Cleaned data set
train <- train[ ,-(1:7)] # Remove vars not measuring anything
train$classe <- as.factor(train$classe)

# Make a train and test set
set.seed(498)

inTrain <- sample(1:nrow(train), (0.6*nrow(train)))

cv.set <- train[-inTrain, ]
train <- train[inTrain, ]

######### Remove outliers from training set ##########
trainOutlier <- scale(train[ ,-53])
apply(trainOutlier, 2, mean) # All zero

# Training set with outliers (> 2 std) removed
trainOutRemove <- train[ ,-53]
trainOutRemove[abs(trainOutlier) > 2.56] <- NA
trainOutRemove <- cbind(trainOutRemove, train$classe)
colnames(trainOutRemove)[53] <- "classe"

# Mediaan invullen voor NA
trainOutRemove2 <- trainOutRemove[ ,-53]

for (j in 1:ncol(trainOutRemove2))
  {
   trainOutRemove2[ ,j][is.na(trainOutRemove2[ ,j])] <- median(trainOutRemove2[ ,j], na.rm=TRUE)
  }

trainOutRemove2 <- cbind(trainOutRemove2, trainOutRemove$classe)
colnames(trainOutRemove2)[53] <- "classe"


############# Random forest with randomForest package #########3
library(randomForest)

########### Random Forest 1 (500 trees, mtry=10)#############
set.seed(701)

rf1 <- randomForest(classe ~ ., data=train, ntree=500, mtry=10)
print(rf1) # OOB error rate: 0.65%

# Performance on cv set
predsRF1 <- predict(object=rf1, newdata=cv.set, type="response")
table(cv.set$classe, predsRF1) # Miss-class 0.64%
(1-(sum(predsRF1==cv.set$classe)) / nrow(cv.set)) * 100


########### Random Forest 2 (1250 trees, mtry=10)#############
set.seed(701)

rf2 <- randomForest(classe ~ ., data=train, ntree=1250, mtry=10)
print(rf2) # OOB error rate: 0.66%

# Performance on cv set
predsRF2 <- predict(object=rf2, newdata=cv.set, type="response")
table(cv.set$classe, predsRF2) # Miss-class 0.62%
(1-(sum(predsRF2==cv.set$classe)) / nrow(cv.set)) * 100


########### Random Forest 3 (500 trees, mtry=default=7)#############
set.seed(701)

rf3 <- randomForest(classe ~ ., data=train, ntree=500)
print(rf3) # OOB error rate: 0.67%

# Performance on cv set
predsRF3 <- predict(object=rf3, newdata=cv.set, type="response")
table(cv.set$classe, predsRF3) # Miss-class 0.66%
(1-(sum(predsRF3==cv.set$classe)) / nrow(cv.set)) * 100


########### Random Forest 4 (1250 trees, mtry=10, outliers removed)#########
set.seed(701)

rf4 <- randomForest(classe ~ ., data=trainOutRemove, ntree=1250, 
  mtry=10, na.action=na.omit)
print(rf4) # OOB error rate: 0.99%

# Performance on cv set
predsRF4 <- predict(object=rf4, newdata=cv.set, type="response")
table(cv.set$classe, predsRF4) # Miss-class 6.4% TERRIBLE !!!!!!
(1-(sum(predsRF4==cv.set$classe)) / nrow(cv.set)) * 100


########### Random Forest 5 (1250 trees, mtry=10, top20 vars)#########
# Random forest 2 presteert beste op cv set,top 20 destilleren
importanceTable <- as.data.frame(importance(rf2))
importanceTable <- cbind(rownames(importanceTable), importanceTable)
importanceTable <- importanceTable[order(-importanceTable[ ,2]), ]

train.top20 <- train[ ,names(train) %in% importanceTable[1:20,1]]
train.top20 <- cbind(train.top20, train$classe)
colnames(train.top20)[21] <- "classe"

rf5 <- randomForest(classe ~ ., data=train.top20, ntree=1250, mtry=10)
print(rf5) # OOB error rate: 1.04%

# Performance on cv set
predsRF5 <- predict(object=rf5, newdata=cv.set, type="response")
table(cv.set$classe, predsRF5) # Miss-class 0.99% Not so good......
(1-(sum(predsRF5==cv.set$classe)) / nrow(cv.set)) * 100


########### Random Forest 6 (1250 trees, mtry=10, median for outliers)#######
set.seed(701)

rf6 <- randomForest(classe ~ ., data=trainOutRemove2, ntree=1250, mtry=10)
print(rf6) # OOB error rate: 0.70%

# Performance on cv set
predsRF6 <- predict(object=rf6, newdata=cv.set, type="response")
table(cv.set$classe, predsRF6) # Miss-class 0.57%
(1-(sum(predsRF6==cv.set$classe)) / nrow(cv.set)) * 100

# Random Forest 6 has best performance on cv set
# Cross validate random forest 6
# Duurt ongeveer half uur !!!!
cv.rf6 <- rfcv(trainOutRemove2, trainOutRemove2[ ,53], cv.fold=10)

with(cv.rf6, plot(n.var, error.cv, log="x", type="o", lwd=2))

########### Random Forest 7 (1250 trees, mtry=10, median for outliers, top 20) #######
importTable <- as.data.frame(importance(rf6))
importTable <- cbind(rownames(importTable), importTable)
importTable <- importTable[order(-importTable[ ,2]), ]

# RF7 with top 20 important vars
train.top20 <- trainOutRemove2[ ,names(trainOutRemove2) %in% importTable[1:20,1]]
train.top20 <- cbind(train.top20, trainOutRemove2$classe)
colnames(train.top20)[21] <- "classe"

rf7 <- randomForest(classe ~ ., data=train.top20, ntree=1250, mtry=10)
print(rf7) # OOB error rate: 0.93%

# Performance on cv set
predsRF7 <- predict(object=rf7, newdata=cv.set, type="response")
table(cv.set$classe, predsRF7) # Miss-class 0.93%
(1-(sum(predsRF7==cv.set$classe)) / nrow(cv.set)) * 100


############ Testing rf6 on the 20 pml test set ############
testPML <- read.csv("H:\\Coursera\\Data Science Track\\08_Practical Machine Learning\\week 3\\Project 1\\pml-testing.csv",
                  header=TRUE, stringsAsFactors=FALSE, sep=",", na.strings="NA")

testPML <- testPML[ ,-remove]
testPML <- testPML[ ,-(1:7)]
testPML <- testPML[ ,-(53)]

preds.PML.RF6 <- predict(object=rf6, newdata=testPML, type="response")

