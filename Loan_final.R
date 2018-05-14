library(e1071)
library(caret)
library(ROSE)
library(pROC)
require(party)
library(rpart)
library(rpart.plot)
library(pscl)
library(ggplot2)
library(randomForest)


#Read data
loans  <- read.csv("C:/Users/zmy/Documents/RPI/ITWS 6600 Data/A6/loan_data.csv")
summary(loans)

# Distribution
par(mfrow=c(2,2))
barplot(table(loans$credit.policy),main="credit policy")
barplot(table(factor(loans$inq.last.6mths)),main="inquiries in last 6 months.")
barplot(table(factor(loans$pub.rec)),main="public records ")
barplot(table(loans$not.fully.paid),main="not fully paid")

par(mfrow=c(1,1))
pie(table(loans$purpose),main="Purpose",radius=1)

par(mfrow=c(2,2))
hist(loans$int.rate, breaks = 20, xlab = "int.rate",main = "interest rate")
hist(loans$installment , breaks = 20, xlab = " installment ",main = " installment ")
hist(loans$log.annual.inc , breaks = 20, xlab = "log.annual.inc ",main = "annual income ")
hist(loans$dti, breaks = 20, xlab = " dti",main = "debt-to-income ratio")

par(mfrow=c(2,2))
hist(loans$fico, breaks = 20, xlab = "fico",main = "FICO credit score ")
hist(loans$days.with.cr.line, breaks = 20, xlab = "days.with.cr.line",main = "days.with.cr.line")
hist(loans$revol.bal, breaks = 20, xlab = "revol.bal",main = "revolving balance")
hist(loans$revol.util, breaks = 20, xlab = "revol.util",main = " revolving line utilization rate")

par(mfrow=c(2,2))
boxplot(loans$int.rate, main = "interest rate")
boxplot(loans$log.annual.inc ,main = "annual income ")
boxplot(loans$days.with.cr.line, main = "days.with.cr.line")
boxplot(loans$revol.bal,main = "revolving balance")


#correlation
library(corrplot)
res=cor(subset(loans,select=c(-2)))
par(mfrow=c(1,1))
corrplot(res, type = "upper", order = "hclust", tl.col = "black")

ggplot(loans, aes(factor(purpose))) + geom_bar(aes(fill=not.fully.paid), position='dodge') + theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggplot(loans, aes(int.rate, fico)) + geom_point(aes(color=not.fully.paid), alpha=0.5) + theme_bw()
ggplot(loans, aes(revol.util,fico)) + geom_point(aes(color=not.fully.paid), alpha=0.5) + theme_bw()


#Data cleaning
loans$credit.policy <- as.factor(loans$credit.policy)
loans$not.fully.paid <- as.factor(loans$not.fully.paid)
summary(loans)


#Split data
set.seed(3456)
trainIndex <- createDataPartition(loans$not.fully.paid, p = .7, 
                                  list = FALSE, 
                                  times = 1)
loansTrain <- loans[ trainIndex,]
loansTest  <- loans[-trainIndex,]


# Deal with unbalanced class
prop.table(table(loansTrain$not.fully.paid))
loansTrain<- ROSE(not.fully.paid ~ ., data = loansTrain, seed = 1)$data
table(loansTrain$not.fully.paid)


#Model 1-Logistics Regression
logit = glm(not.fully.paid ~ .,data=loansTrain, family = "binomial"(link='logit'))
summary(logit)
anova(logit, test="Chisq")
pR2(logit)

# threshold
pred.logit <- predict(logit,newdata=loansTest[-14],type="response")
roc.logit=roc(loansTest$not.fully.paid,pred.logit)
plot(roc.logit,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("red","blue"),max.auc.polygon=TRUE,auc.polygon.col="chocolate1",print.thres=TRUE)

#predict
pred.logit <- ifelse(pred.logit > 0.6,1,0)

#Evaluation
roc.curve(loansTest$not.fully.paid, pred.logit)
table(pred.logit,loansTest$not.fully.paid,dnn=c("Predicted","Actual"))
confusionMatrix(table(pred.logit,loansTest$not.fully.paid))


#Model2- SVM
svm1 <- svm(not.fully.paid~ ., data=loansTrain,kernel="radial",cross=5,type="C-classification")
summary(svm1)
pred.svm1 <- predict(svm1, loansTest[-14])


#Tuning
tuned1<-tune.svm(not.fully.paid~credit.policy+purpose+int.rate+installment+
                   log.annual.inc+fico+inq.last.6mths+pub.rec+revol.util+revol.bal, 
                 data=loansTrain,gamma = 0.1*(1:3),cost = 10^(0:2))
summary(tuned1)


svm2 <- svm(not.fully.paid~credit.policy+purpose+int.rate+installment+
              log.annual.inc+fico+inq.last.6mths+pub.rec+revol.util+revol.bal, 
            data=loansTrain,kernel="radial",cross=10,type="C-classification", cost =1, gamma=0.3)
summary(svm2)
pred.svm2 <- predict(svm2, newdata=subset(loansTest,select=c(1,2,3,4,5,7,9,10,11,13)))


#Evaluation
roc.svm=roc(loansTest$not.fully.paid,as.numeric(pred.svm2))
plot(roc.svm,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("red","blue"),max.auc.polygon=TRUE,auc.polygon.col="chocolate1",print.thres=TRUE)
table(pred.svm2,loansTest$not.fully.paid,dnn=c("Predicted","Actual"))
confusionMatrix(table(pred.svm2,loansTest$not.fully.paid))



# Model 3 CART

loans$revol.bal<-log(loans$revol.bal)
#Split data
set.seed(3456)
trainIndex <- createDataPartition(loans$not.fully.paid, p = .7, 
                                  list = FALSE, 
                                  times = 1)
loansTrain <- loans[ trainIndex,]
loansTest  <- loans[-trainIndex,]


# Deal with unbalanced class
prop.table(table(loansTrain$not.fully.paid))
loansTrain<- ROSE(not.fully.paid ~ ., data = loansTrain, seed = 1)$data
table(loansTrain$not.fully.paid)

#Train model with 10-fold crossvalidation
cart <- rpart(not.fully.paid ~ ., data = loansTrain, xval=10)
summary(cart)
prp(cart, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,main="Classification Tree")
printcp(cart)
plotcp(cart)

# Find threshold
pred.cart<- predict(cart, newdata=loansTest[-14])
roc.cart<-roc(loansTest$not.fully.paid, pred.cart[,2])
plot(roc.cart,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("red","blue"),max.auc.polygon=TRUE,auc.polygon.col="chocolate1",print.thres=TRUE)

#Prune tree
pcart<- prune(cart, cp=0.02345)
prp(pcart, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,main="Pruned Classification Tree")

#Predict
pred.pcart<- predict(pcart,newdata = loansTest[-14])
pred.pcart<- ifelse(pred.pcart[,2] > 0.609,1,0)

#Evaluation
roc.curve(loansTest$not.fully.paid, pred.pcart)
table(pred.pcart,loansTest$not.fully.paid,dnn=c("Predicted","Actual"))
confusionMatrix(table(pred.pcart,loansTest$not.fully.paid))


#Model 4- Random forest
#Train the model
rf <- randomForest(not.fully.paid ~ ., data = loansTrain, importance = TRUE)
rf
varImpPlot(rf,type=2)

# Prediction
rf.pred<-predict(rf,loansTest)
confusionMatrix(rf.pred,loansTest$not.fully.paid)
roc.rf=roc(loansTest$not.fully.paid,as.numeric(rf.pred))
plot(roc.rf,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("red","blue"),max.auc.polygon=TRUE,auc.polygon.col="chocolate1",print.thres=TRUE)



