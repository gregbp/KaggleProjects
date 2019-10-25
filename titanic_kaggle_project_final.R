"This code was created for the Kaggle competion 'Titanic: Machine Learning from Disaster'
See more details here -> https://www.kaggle.com/c/titanic/overview
"

"Three approaches were implemented; Logistic Regression, Decision Trees and Random Forrest.
Logistic Regression and Random Forrest scored 0.75119/1 accuracy
and Decision Trees scored 0.73684/1 accuracy. 
Finally, the results of Random Forrest were submitted to Kaggle.
"

library(dplyr)
library(ggplot2)
library(caTools)
library(ISLR)
library(class)
library(rpart) 
library(randomForest)
library(rpart.plot)

"There are passengers whose age is not available. For this reason, ages were imputed by the ticket class. 
The imputed age is the mean of the ages of every ticket class."
impute_age <- function(age,class){
  out <- age
  for (i in 1:length(age)){
    
    if (is.na(age[i])){
      
      if (class[i] == 1){
        out[i] <- 37
        
      }else if (class[i] == 2){
        out[i] <- 29
        
      }else{
        out[i] <- 24
      }
    }else{
      out[i]<-age[i]
    }
  }
  return(out)
}



          #Train and Test set creation
train = read.csv("C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\train.csv")
test = read.csv("C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\test.csv") 

fixed.ages <- impute_age(train$Age,train$Pclass)
train$Age <- fixed.ages

fixed.ages <- impute_age(test$Age,test$Pclass)
test$Age <- fixed.ages

train1 = select(train, PassengerId, Pclass, Sex, Age, Survived)
test1 = select(test, PassengerId, Pclass, Sex, Age)
 
 
sum(is.na(test1))






            #Logistic Regression

log.model <- glm(formula=Survived ~ . , family=binomial(link='logit'),data=train1)


fitted.probabilities <- predict(log.model,newdata=test1,type='response')
fitted.results <- ifelse(fitted.probabilities > 0.5,1,0)


PassengerId <- test1$PassengerId
Survived <- fitted.results
dffr <- data.frame(PassengerId,Survived)
 
head(dffr)
sum(is.na(dffr))
 

write.table('PassengerId,Survived',"C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\lr_res.csv" ,row.names = FALSE,col.names=FALSE,sep=",",quote = FALSE)
write.table(dffr,"C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\lr_res.csv",row.names = FALSE,col.names=FALSE, sep=",", append = T)
 






              #Trees

 
cat( '\n\n\tDecision Tree\n')
 
 
#Decision Tree
tree <- rpart(Survived ~.,method='class',data = train1)
tree.preds <- as.data.frame(predict(tree,test1))
print(head(tree.preds))


tree.preds$Survived <- ifelse(tree.preds$'1' > 0.5,1,0)
print(head(tree.preds))

 
#prp(tree)

PassengerId<-test1$PassengerId
dffr <-data.frame(PassengerId,tree.preds$Survived)
head(dffr)
sum(is.na(dffr))

write.table('PassengerId,Survived',"C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\dt_res.csv" ,row.names = FALSE,col.names=FALSE,sep=",",quote = FALSE)
write.table(dffr,"C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\dt_res.csv",row.names = FALSE,col.names=FALSE, sep=",", append = T)







              #Random Forrest
cat( '\n\n\tRandom Forrest\n')

rf.model <- randomForest(Survived ~ . , data = train1,importance = TRUE)

 

tree.preds2 <- as.data.frame(predict(rf.model,test1))
#print(head(tree.preds2))
tree.preds2$Survived <- ifelse(tree.preds2 > 0.5,1,0)
#print(head(tree.preds2))
#print(head(tree.preds2$Survived))

PassengerId<-test1$PassengerId
dffr <-data.frame(PassengerId,tree.preds2$Survived)
head(dffr)
sum(is.na(dffr))

write.table('PassengerId,Survived',"C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\rf_res.csv" ,row.names = FALSE,col.names=FALSE,sep=",",quote = FALSE)
write.table(dffr,"C:\\Users\\grigo\\Documents\\R udemy solutions\\titanic project\\titanic\\rf_res.csv",row.names = FALSE,col.names=FALSE, sep=",", append = T)
 













