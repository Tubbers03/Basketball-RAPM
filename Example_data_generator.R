# Nstint = 100
# alpha = sort(rnorm(15,mean = 0,sd = .8),decreasing = TRUE)
# probs = exp(alpha)/sum(exp(alpha))
# counts = numeric(15)
# for(i in 1:Nstint){
#   these = sample(1:15,size=5,prob = probs)
#   players = numeric(15)
#   players[these] =1
#   counts = counts + players
# }
# 
# p = counts/Nstint
# matrix(round(p * 48),ncol=5,byrow = TRUE)

#_________________________________
# Install Packages
install.packages("BART")
install.packages("glmnet")
#_________________________________

#_________________________________
# Library
library(glmnet)
library(BART)
#_________________________________

#_________________________________
# Setup
setwd("/Users/aricalley/Documents/Research Project")
getwd()
list.files()
#_________________________________

#_________________________________
train <- read.csv("bart_train.csv")
test <- read.csv("bart_test.csv")

X_train <- as.matrix(train[, -ncol(train)])
y_train <- train$Y

X_test <- as.matrix(test[, -ncol(test)])
y_test <- test$Y
#_________________________________

#_________________________________
#BART
bart_model <- wbart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test
)

bart_predictions <- bart_model$yhat.test.mean

bart_rmse <- sqrt(mean((y_test - bart_predictions)^2))

print(bart_rmse)
#_________________________________

#_________________________________
#OLS
ols_model <- lm(y_train ~ X_train - 1)

ols_predictions <- X_test %*% coef(ols_model)

ols_rmse <- sqrt(mean((y_test - ols_predictions)^2))
#_________________________________

#_________________________________
#RIDGE
ridge_model <- glmnet(
  X_train,
  y_train,
  alpha = 0,
  intercept = FALSE
)

ridge_predictions <- predict(
  ridge_model,
  newx = X_test,
  s = 10
)

ridge_rmse <- sqrt(mean((y_test - ridge_predictions)^2))

ridge_rmse
#_________________________________

#_________________________________
#ALL 3
comparison <- data.frame(
  Model = c("OLS", "Ridge", "BART"),
  RMSE = c(
    ols_rmse,
    ridge_rmse,
    bart_rmse
  )
)

comparison
#_________________________________
