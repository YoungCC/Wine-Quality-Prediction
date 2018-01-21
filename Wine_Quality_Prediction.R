# ===================================================================================================
# Title: Wine Quality Prediction
# Author: Yang Cao
# Date: November 30, 2017
# Purpose: Predict the quality ratings from various physicochemical properties of red and white wines.
# ===================================================================================================

rm(list = ls())

library(caret)
library(corrplot)
library(kknn)
library(randomForest)
library(kernlab)
library(ggplot2)
library(GGally)

# Read data files from the UC Irvine Machine Learning Repository Website
red_wine <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", header = TRUE, sep = ";")
white_wine <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", header = TRUE, sep = ";")
View(red_wine)
View(white_wine)

# Data preprocessing: merge two objects into one
red_wine['color'] <- "red"
white_wine['color'] <- "white"
wine <- merge(red_wine, white_wine, all = TRUE)
attach(wine)

# Summary of the data set
dim(wine)
names(wine)
head(wine)
View(wine)
str(wine)
summary(wine)

length(quality)
mean(quality) 
min(quality)
max(quality)
sd(quality)
summary(quality)
table(quality)
quality <- as.factor(quality)
is.factor(quality)
plot(quality)
quality <- as.vector(quality)
plot(quality)
typeof(quality)
qplot(quality, data = wine, fill = color, binwidth = 1, main = "Quality Distribution of Red and White Wine") +
  scale_x_continuous(breaks = seq(2,10,1), lim = c(2,10)) +
  scale_y_sqrt() + xlab('Quality') + ylab('Quantity')
quality <- as.numeric(quality)
typeof(quality)

summary(color)
table(color)

# Exploratory Data Analysis
variables <- c("fixed.acidity", "volatile.acidity", "citric.acid",
                "residual.sugar", "chlorides", "free.sulfur.dioxide",
                "total.sulfur.dioxide", "density", "pH", "sulphates",
                "alcohol", "quality", "color")

# EDA: Histograms of each variable
par(mfrow=c(3,4))
for( i in seq(1,12)){
  hist(wine[,i], col = "pink", xlab = variables[i], ylab = "Count", main = paste(variables[i], "distribution",sep = " "))
}

for( i in seq(1,12)){
  hist(red_wine[,i], col = "firebrick", xlab = variables[i], ylab = "Count", main = paste(variables[i], "distribution for Red wine",sep = " "))
}

for( i in seq(1,12)){
  hist(white_wine[,i], col = "light yellow", xlab = variables[i], ylab = "Count", main = paste(variables[i], "distribution for White wine",sep = " "))
}

# EDA: Boxplots of each variable
for( i in seq(1,12)){
  boxplot(wine[,i], col = "pink", main = variables[i])
}

for( i in seq(1,12)){
  boxplot(red_wine[,i], col = "firebrick", main = paste(variables[i], "of Red wine",sep = " "))
}

for( i in seq(1,12)){
  boxplot(white_wine[,i], col = "light yellow", main = paste(variables[i], "of White wine",sep = " "))
}

# EDA: Scatterplots of target and each predictor (Using the white wine dataset)
for( i in seq(1,12)){
  plot(white_wine[,i], white_wine[,12], xlab = variables[i], ylab="quality", main = paste("The relationship between", variables[i], "and quality",sep = " "))
  abline(lm(white_wine[,12] ~ white_wine[,i]), lty = 2, lwd=3, col="firebrick")
}
par(mfrow=c(1,1))

max.sug <- which(white_wine$residual.sugar == max(white_wine$residual.sugar))
white_wine <- white_wine[-max.sug, ]
max.free <- which(white_wine$free.sulfur.dioxide == max(white_wine$free.sulfur.dioxide))
white_wine <- white_wine[-max.free, ]

white_wine <- subset(white_wine, select = -c(color))
pairs(white_wine)
ggpairs(white_wine)

cor.white_wine <- cor(white_wine)
corrplot(cor.white_wine, method = 'number')

summary(subset(white_wine, select = -quality))
cor(subset(white_wine, select = -quality))
cor(white_wine[,-12], method="spearman")

# Create train and test sets (Using the white wine dataset)
set.seed(42)
N <- nrow(white_wine)
train <- sample(N, 3*N/4)
whiteTrain <- white_wine[train,]
whiteTest  <- white_wine[-train,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

# Feature Selection (Using the white wine dataset)
detach(wine)
attach(white_wine)
fit <- step(lm(whiteTrain$quality ~ 1, whiteTrain), scope=list(lower=~1,  
    upper = ~fixed.acidity + volatile.acidity + citric.acid + residual.sugar + 
    chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol), direction="forward")

# Fit Multiple Linear Regression Model (Using the white wine dataset)
model.multiple <- lm(quality ~ alcohol + volatile.acidity + residual.sugar + density + 
    pH + sulphates + fixed.acidity + free.sulfur.dioxide , data = whiteTrain)
summary(model.multiple)
print(model.multiple)
confint(model.multiple)
library(car)
vif(model.multiple)

par(mfrow=c(2,2))
plot(model.multiple)
par(mfrow=c(1,1))

pred.train <- predict(model.multiple, data = whiteTrain, interval = "confidence")
pred.test <- predict(model.multiple, newdata = whiteTest)
mse.multiple <- mean((whiteTest$quality - pred.test)^2)
mse.multiple

# Fit Polynomial Regression (Using the white wine dataset)
model.polylinear <- lm(quality ~ poly(alcohol,2) + poly(volatile.acidity,2) + poly(residual.sugar,4) + 
                         poly(free.sulfur.dioxide,5) + poly(fixed.acidity,2) + sulphates + poly(density,3) + poly(pH,2), data=whiteTrain)
summary(model.polylinear)
confint(model.polylinear)
vif(model.polylinear)
residualPlots(model.polylinear, pch=19, col="pink", cex=0.6)

pred.train <- predict(model.polylinear, data = whiteTrain, interval = "confidence")
pred.test <- predict(model.polylinear, newdata = whiteTest)
mse.polylinear <- mean((whiteTest$quality - pred.test)^2)
mse.polylinear

free.sulfur.dioxide.grid <- seq(from = min(free.sulfur.dioxide), to = max(free.sulfur.dioxide))
free.sulfur.dioxide.grid
length(free.sulfur.dioxide.grid)

plot(free.sulfur.dioxide, white_wine$quality, cex=.5, col="darkgray", main="Degree-4Polynomial")
colors <- c("red", "blue", "green", "black")
for (i in seq(1,4)){
  fit <- lm(white_wine$quality ~ poly(free.sulfur.dioxide,i))
  pred <- predict(fit, newdata = list(free.sulfur.dioxide = free.sulfur.dioxide.grid), se = TRUE)
  se.bands <- cbind(pred$fit + 2*pred$se.fit, pred$fit - 2*pred$se.fit)
  lines(free.sulfur.dioxide.grid, pred$fit, lwd=2, col=colors[i])
  matlines(free.sulfur.dioxide.grid, se.bands, lwd=1, col=colors[i], lty=3)
}

# Generalized Additive Models (Using the white wine dataset)
gam1 <- lm(white_wine$quality ~ ns(alcohol, 2) + ns(volatile.acidity,5)  + ns(residual.sugar,4) + 
             ns(free.sulfur.dioxide,6) + ns(fixed.acidity,2) + ns(sulphates,1) + 
             ns(density,3) + ns(pH,3))
summary(gam1)
par(mfrow = c(2,2))
plot(gam1)

pred.train <- predict(gam1, data = whiteTrain, interval = "confidence")
pred.test <- predict(gam1, newdata = whiteTest)
mse.gam1 <- mean((whiteTest$quality - pred.test)^2)
mse.gam1

library(splines)
library(foreach)
library(gam)
gam2 <- gam(white_wine$quality ~ s(alcohol, 2) + s(volatile.acidity,5)  + s(residual.sugar,4) + 
              s(free.sulfur.dioxide,6) + s(fixed.acidity,2) + s(sulphates,1) + 
              s(density,3) + s(pH,3))
summary(gam2)

pred.train <- predict(gam2, data = whiteTrain, interval = "confidence")
pred.test <- predict(gam2, newdata = whiteTest)
mse.gam2 <- mean((whiteTest$quality - pred.test)^2)
mse.gam2

# Regression Tree Method
library(rpart)
library(plotly) 
library(rpart.plot)

m.rpart <- rpart(quality ~., data = whiteTrain)
summary(m.rpart)
par(mfrow = c(1,1))
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE)

pred.train <- predict(m.rpart, data = whiteTrain, interval = "confidence")
pred.test <- predict(m.rpart, newdata = whiteTest)
mse.rpart <- mean((whiteTest$quality - pred.test)^2)
mse.rpart

# Random Forest (Regression)
library(randomForest)
model.rf <- randomForest(white_wine$quality[train] ~ . - quality, data = whiteTrain, importance=TRUE, do.trace=100)
model.rf
getTree(model.rf, 1, labelVar=TRUE)
head(model.rf$importanceSD)
rf.pred <- predict(model.rf, newdata = whiteTest)
mse.rf <- mean((whiteTest$quality - rf.pred)^2)
mse.rf
table(rf.pred, white_wine[-train,]$quality)

# Random Forest (Classification)
white_wine$taste <- ifelse(white_wine$quality < 6, 'bad', 'good')
white_wine$taste[white_wine$quality == 6] <- 'normal'
white_wine$taste <- as.factor(white_wine$taste)
table(white_wine$taste)
library(randomForest)
model.rf <- randomForest(white_wine$taste[train] ~ . - quality, data = whiteTrain, importance=TRUE, do.trace=100)
model.rf
model.rf$classes
head(model.rf$predicted)
table(model.rf$predicted == white_wine[train,]$taste)
head(model.rf$err.rate)
head(model.rf$oob.times)
head(model.rf$importanceSD)
model.rf$localImportance
rf.pred <- predict(model.rf, newdata = whiteTest)
table(rf.pred, white_wine[-train,]$taste)

# KNN
set.seed(2)
model.knn <- train(whiteTrain$quality ~., data = whiteTrain, method = "knn", tuneGrid = data.frame(.k = 1:30), trControl = fitControl)
knnPred <- predict(model.knn, newdata = whiteTest)
r2_knn <- R2(knnPred, whiteTest$quality)
rmse_knn <- RMSE(knnPred, whiteTest$quality)

# Neural Networks
set.seed(4)
model.nn <- train(whiteTrain$quality ~., data = whiteTrain, method = "nnet", linout = TRUE, maxit = 100, tuneGrid = expand.grid(.size=c(1:5), .decay=c(0,0.001,0.01,0.1)), trControl = fitControl)
NNPred <- predict(model.nn, newdata = whiteTest)
r2_nn <- R2(NNPred, whiteTest$quality)
rmse_nn <- RMSE(NNPred, whiteTest$quality)

# SVM
set.seed(7)
model.svm <- train(whiteTrain$quality ~., data = whiteTrain, method = "svmRadial",  tuneLength = 5, trControl = fitControl)
SVMPred <- predict(model.svm, newdata = whiteTest)
r2_svm <- R2(SVMPred, whiteTest$quality)
rmse_svm <- RMSE(SVMPred, whiteTest$quality)

# Splines (Using "free.sulfur.dioxide" as the predictor for exploration)
library(splines)

knot.position <- c(summary(free.sulfur.dioxide))
fit <- lm(quality ~ bs(free.sulfur.dioxide, knots=knot.position), data=white_wine)
summary(fit)
pred <- predict(fit, newdata=list(free.sulfur.dioxide=free.sulfur.dioxide.grid), se=TRUE)
plot(free.sulfur.dioxide, white_wine$quality, cex=.5, col="darkgray")
lines(free.sulfur.dioxide.grid, pred$fit, lwd=2, col="red")
lines(free.sulfur.dioxide.grid, pred$fit + 2*pred$se, lty="dashed", col="red")
lines(free.sulfur.dioxide.grid, pred$fit - 2*pred$se, lty="dashed", col="red")
abline(v = knot.position, lty="dashed")

# Smoothing Splines (Using "pH" as the predictor for exploration)
fit <- smooth.spline(pH, white_wine$quality, cv=TRUE)
summary(fit)
fit
plot(pH, white_wine$quality, cex=.5, col="darkgray")
lines(fit,  col="red", lwd=2)
legend("topright", legend=c("6.3 DF (LOOCV)"), col=c("red"), lty=1, lwd=2, cex=.8)

# Logistic regression with multiple predictors
model.logic <- glm(I(quality>5) ~ .-quality, data = whiteTrain, family = binomial)
summary(model.logic)
pred.probs <- predict (model.logic, whiteTest, type = "response")
pred.quality <- rep("FALSE", N/2)
pred.quality[pred.probs > 0.5] <- "TRUE"

confusion.matrix <- table(I(quality[-train]>5), pred.quality)
addmargins(confusion.matrix)

error.rate <- mean(pred.quality != I(quality[-train]>5))
error.rate

library(boot)
model.logic2 <- glm(I(quality>5) ~ fixed.acidity + volatile.acidity + citric.acid + 
                      residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide +
                      density + pH + sulphates + alcohol, family = binomial)

cost <- function(r, pi = 0) {
  mean(abs(r-pi) > 0.5)
}

cv.error.10 <- cv.glm(white_wine, model.logic2, cost, K=10)
cv.error.10$delta

# Polynomial Logistic Regression
model.polylogic <- glm(I(quality > 5) ~ poly(free.sulfur.dioxide, 4), family=binomial)
summary(model.polylogic)
pred <- predict(model.polylogic, newdata = list(free.sulfur.dioxide = free.sulfur.dioxide.grid), se = TRUE)
pred2 <-predict(model.polylogic, newdata = list(free.sulfur.dioxide = free.sulfur.dioxide.grid), type="response")
pfit <- exp(pred$fit) / (1+ exp(pred$fit))
se.bands.logit <- cbind(pred$fit + 2*pred$se.fit, pred$fit - 2*pred$se.fit)
se.bands <- exp(se.bands.logit) / (1 + exp(se.bands.logit))

plot(free.sulfur.dioxide, I(quality>5), type="n", ylim=c(0,1))
lines(free.sulfur.dioxide.grid, pfit, lwd=2, col="blue")
matlines(free.sulfur.dioxide.grid, se.bands, lwd=1, col="blue", lty=3)
rug(jitter(free.sulfur.dioxide[quality<=5]), side = 1, ticksize = 0.02)
rug(jitter(free.sulfur.dioxide[quality>5]), side = 3, ticksize = 0.02)

plot(free.sulfur.dioxide, I(quality>5), type="n", ylim=c(0,1), main="Degree-4Polynomial")
colors <- c("red", "blue", "green", "black")
rug(jitter(free.sulfur.dioxide[quality<=5]), side = 1, ticksize = 0.02)
rug(jitter(free.sulfur.dioxide[quality>5]), side = 3, ticksize = 0.02)
for (i in seq(1,4)){
  fit <- glm(I(quality > 5) ~ poly(free.sulfur.dioxide, i), family=binomial)
  pred <- predict(fit, newdata = list(free.sulfur.dioxide = free.sulfur.dioxide.grid), se = TRUE)
  pfit <- exp(pred$fit) / (1+ exp(pred$fit))
  se.bands.logit <- cbind(pred$fit + 2*pred$se.fit, pred$fit - 2*pred$se.fit)
  se.bands <- exp(se.bands.logit) / (1 + exp(se.bands.logit))
  lines(free.sulfur.dioxide.grid, pfit, lwd=2, col=colors[i])
  matlines(free.sulfur.dioxide.grid, se.bands, lwd=1, col=colors[i], lty=3)
}
