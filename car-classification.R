rm(list =ls())
library(e1071)
library(glmnet)


path = "../car.txt"

readData = function(path){
  raw = read.table(path, sep = ",")
  data = data.frame(raw[, 1:6], eval = ifelse(raw[, 7] == "unacc", "Negative", "Positive"))
  data$binary = ifelse(data$eval=="Positive",1,0)
  return(data)}

splitData = function(data){
  trainInd = sample(1:nrow(data), nrow(data)/2, replace = F)
  train = data[trainInd,]
  test = data[-trainInd,]
  return(list(train, test))
}

evaluation = function(train, test){
  
  # Preprocess the data for compatibility with the glmnet package.
  x = model.matrix(~., train[,1:6])[,-1]
  xTest = model.matrix(~., test[,1:6])[,-1]
  y = as.numeric(train[,8])
  yTest = as.numeric(test[,8])
  
  # Initialize accuracy Matrix
  nrModels = 6
  N = nrow(train)
  accuracy = matrix(NA, nrModels, N)
  
  for (n in 1:N){
    
    # Train NB with Laplace Smoothing, s.t. a = 1
    
    bayesCl = naiveBayes(eval ~ ., data = train[1:n, !(names(train) %in% c("binary"))], laplace = 1)
    predCl = predict(bayesCl, test[,1:6])
    accuracy[1, n] = mean(predCl == test$eval)
    
    # Train NB with Laplace Smoothing, s.t. a = 0.1 - closer to ML estimation
    
    bayesClMLE = naiveBayes(eval ~ ., data = train[1:n, !(names(train) %in% c("binary"))], laplace = 0.1)
    predClMLE = predict(bayesClMLE, test[,1:6])
    accuracy[2, n] = mean(predClMLE == test$eval)
    
    # Train NB with Laplace Smoothing, s.t. a = 10
    
    bayesClA = naiveBayes(eval ~ ., data = train[1:n, !(names(train) %in% c("binary"))], laplace = 10)
    predClA = predict(bayesClA, test[,1:6])
    accuracy[3, n] = mean(predClA == test$eval)
    
    # Train full Logistic Regression Model
    
    tryCatch({
      logReg = glm(binary~., data = train[1:n, !(names(train) %in% c("eval"))], family = binomial(), control = list(maxit = 100))
      predLR = ifelse(predict(logReg, test[,1:6], type = "response")>0.5, 1, 0)
      accuracy[4, n] = mean(test$binary == predLR)
      
    }, error = function(e){})
    
    # Train reduced Logistic Regression Model
    
    tryCatch({
      logRegRed = glm(binary~., data = train[1:n,(names(train) %in% c("binary", "V1", "V2"))], family = binomial(), control = list(maxit = 100))
      predLRR =  ifelse(predict(logRegRed, test[,1:6], type = "response")>0.5, 1, 0)
      accuracy[5, n] = mean(test$binary == predLRR)
      
    }, error = function(e){})
    
    
    # Train penalized Logistic Regression Model with hyper-parameters defined above.
    
    tryCatch({
      cvProcedure = cv.glmnet(x[1:n,], as.factor(y[1:n]), type.measure = "class", nfolds = 5, family = "binomial", alpha = 0, lambda=seq(0,5,0.1))
      predLRRpen = ifelse(predict(cvProcedure, newx = xTest, s="lambda.min", type = "response")>0.5, 1, 0)
      accuracy[6, n] = mean(predLRRpen == yTest)
    }, error=function(e){})
  }
  return(accuracy)
}


repeatEval = function(data, repetitions = 20){
  
  # Initialization of 3d-array to store results
  accus = array(NA, dim = c(6, nrow(data)/2, repetitions ))
  
  for (rep in 1:repetitions){
    
    # Random Split before each replication.
    splits = splitData(data)
    train = splits[[1]]
    test = splits[[2]]
    accus[, , rep] = evaluation(train, test)
    cat("Current repetition is ", rep, " out of the 20 in total.\n")
  }
  return(accus)
}

probExtract = function(train, test){
  
  nrModels = 2
  N = nrow(train)
  probs = matrix(NA, 4*nrModels, N)
  
  for (n in 1:N){
    
    # Train NB with Laplace Smoothing, s.t. a = 1
    bayesCl = naiveBayes(eval ~ ., data = train[1:n, !(names(train) %in% c("binary"))], laplace = 0.1)
    predCl = predict(bayesCl, test[,1:6])
    
    # Get correctly predicted probabilities for each class and compute respective means 
    indCorrect = which(predCl == test$eval)
    temp = cbind(predict(bayesCl, test[,1:6], type = "raw"), class = test$eval)[indCorrect,]
    confZeros = mean(temp[which(temp[,3] == 1), 1])
    confOnes = mean(temp[which(temp[,3] == 2), 2])
    
    indNegative = which(test$eval=="Negative")
    tempNeg = cbind(predict(bayesCl, test[,1:6], type = "raw"), class = test$eval)[indNegative,]
    tempPos =  cbind(predict(bayesCl, test[,1:6], type = "raw"), class = test$eval)[-indNegative,]
    confPos = mean(tempPos[,2])
    confNeg = mean(tempNeg[,1]) 
    
    
    # Append probabilities and performance per sample size n
    probs[1:2, n] = c(confOnes, confZeros)
    probs[3:4, n] = c(confPos, confNeg)
    # Train full Logistic Regression Model.
    
    tryCatch({
      logReg = glm(binary~., data = train[1:n, !(names(train) %in% c("eval"))], family = binomial(), control = list(maxit = 100))
      predLR = ifelse(predict(logReg, test[,1:6], type = "response")>0.5, 1, 0)
      # Get correctly predicted probabilities for each class and compute respective means 
      indCorrect = which(predLR == test$binary)
      temp = cbind(predict(logReg, test[,1:6], type = "response"), class = test$binary)[indCorrect,]
      confOnes = mean(temp[which(temp[,2] == 1), 1])
      confZeros = 1 - mean(temp[which(temp[,2] == 0), 1])
      
      indNegative = which(test$binary ==0)
      tempNeg = cbind(predict(logReg, test[,1:6], type = "response"), class = test$binary)[indNegative,]
      tempPos = cbind(predict(logReg, test[,1:6], type = "response"), class = test$binary)[-indNegative,]
      confPos = mean(tempPos[,1])
      confNeg = 1 - mean(tempNeg[,1])
      
      # Append probabilities and performance per sample size n
      probs[5:6, n] = c(confOnes, confZeros)
      probs[7:8, n] = c(confPos, confNeg)
    }, error = function(e){})
  }
  if (n%%100==0){
    cat("Current Iteration is ", n, ".\n")
  }
  
  return(probs)
}


data = readData(path)
res = repeatEval(data, repetitions  = 20)
save(res, file = "res.RData")
results = rowMeans(res, dims = 2,na.rm = T)


' Plot Error Rate vs Sample Size '

colors = c('dodgerblue', 'aquamarine4', 'darkolivegreen', 'firebrick4', 'gold2', 'darkorchid4')
plot( 1-results[1,],type='l',ylim=c(0, 0.5), col=colors[1],xlab="Sample Size", ylab='Error Rate',frame.plot = F, cex.lab=1.3,lwd=2.2)
for(i in 2:6){
  lines(1 - results[i,], col=colors[i], lwd=2.2)
}
legend(700, 0.5, legend=c("Naive Bayes, a=1", "Naive Bayes, a=0.1", "Naive Bayes, a=10",
                          "Log. Reg., Full", "Log. Reg., Reduced", "Ridge LR"),
       col=colors, lty=1, cex=0.8, y.intersp = 0.9, text.width = 105, lwd=2)

colors = c('dodgerblue', 'aquamarine4', 'darkolivegreen', 'firebrick4', 'gold2', 'darkorchid4')
plot(log(1:n), 1-results[1,],type='l',ylim=c(0, 0.5), col=colors[1],xlab="Sample Size on the Logarithmic Scale", ylab='Error Rate',frame.plot = F, cex.lab=1.3,lwd=2.2)
for(i in 2:6){
  lines(log(1:n),1 - results[i,], col=colors[i], lwd=2.2)
}
legend(log(90), 0.5, legend=c("Naive Bayes, a=1", "Naive Bayes, a=0.1", "Naive Bayes, a=10",
                              "Log. Reg., Full", "Log. Reg., Reduced", "Ridge LR"),
       col=colors, lty=1, cex=0.8, y.intersp = 0.9, text.width = 20, lwd=2)




' Plot Predictive Probabilities vs Sample Size '

splits = splitData(data)
probs = probExtract(splits[[1]], splits[[2]])


colors = c('dodgerblue', 'aquamarine4', 'brown2', 'firebrick4')
plot(probs[1,],type='l',ylim=c(0.5, 1), col=colors[1],xlab="Sample Size", ylab='Mean Predictive Probabilities for True Positives & Negatives',frame.plot = F, cex.lab=1.3,lwd=2.3)
lines(probs[2,], col=colors[2], lwd=2.3)
lines(probs[5,], col=colors[3], lwd=2.3)
lines(probs[6,], col=colors[4], lwd=2.3)
legend(600, 0.62, legend=c("NB, mean probability for class 1", "NB, mean probability for class 0", "LR, mean probability for class 1",
                           "LR, mean probability for class 0"),
       col=colors, lty=1, cex=0.9, y.intersp = 0.9, text.width = 180, lwd=2.2)


plot(probs[3,],type='l',ylim=c(0.4, 1), col=colors[1],xlab="Sample Size", ylab='Mean Predictive Probabilities for Positive & Negative Labels',frame.plot = F, cex.lab=1.3,lwd=2.3)
lines(probs[4,], col=colors[2], lwd=2.3)
lines(probs[7,], col=colors[3], lwd=2.3)
lines(probs[8,], col=colors[4], lwd=2.3)
legend(600, 0.62, legend=c("NB, mean probability for class 1", "NB, mean probability for class 0", "LR, mean probability for class 1",
                           "LR, mean probability for class 0"),
       col=colors, lty=1, cex=0.9, y.intersp = 0.9, text.width = 180, lwd=2.6)




' Linear Predictors Plotted '

train = splits[[1]]
test = splits[[2]]
logReg = glm(binary~., data = train[, !(names(train) %in% c("eval"))], family = binomial(), control = list(maxit = 25))
linPredTrainLR = logReg$linear.predictors
table(train[which(linPredTrainLR < -15), ]$eval)
linPredTestLR = unname(predict(logReg, test[,1:6], type = "link"))
table(test[which(linPredTestLR < -15), ]$eval)


par(mfrow=c(2,1))
par(mar=c(0,5,3,3))
hist(linPredTrainLR , main="" , xlim=c(-80, 20),ylab=" Frequencies", xlab="", ylim=c(0, 28) , xaxt="n", las=1 , col="slateblue1", breaks=100)
abline(v = 13, lty = 2)
abline(v = -9, lty = 2)
legend(-80, 25, legend=c("Linear Predictors on the train set"),
       col="slateblue1", lty=1, cex=0.9, y.intersp = 0.9, text.width = 30, lwd=2.2)
text(x = -14, y = 25, labels = "p=0.0001", adj = NULL,
     pos = NULL, offset = 0.5, vfont = NULL,
     cex = 1, col = NULL)
text(x = 19, y = 25, labels = " p=0.999998", adj = NULL,
     pos = NULL, offset = 0.5, vfont = NULL,
     cex = 1, col = NULL)
par(mar=c(5,5,0,3))
hist(linPredTestLR , main="" , xlim=c(-80, 20), ylab="", xlab="Values of Linear Predictors", ylim=c(28,0) , las=1 , col="tomato3"  , breaks=100)
abline(v = -9, lty = 2)
abline(v = 13, lty = 2)
legend(-80, 20, legend=c("Linear Predictors on the test set"),
       col="tomato3", lty=1, cex=0.9, y.intersp = 0.9, text.width = 30, lwd=2.2)


middleHist = table(raw[trainInd,7][which(linPredTrainLR>-42 & linPredTrainLR< -10)])
lefthist = table(raw[trainInd,7][which(linPredTrainLR<=-42) ])
righthist = table(raw[trainInd,7][which(linPredTrainLR>-10) ])
