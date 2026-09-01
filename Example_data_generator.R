Nstint = 100
alpha = sort(rnorm(15,mean = 0,sd = .8),decreasing = TRUE)
probs = exp(alpha)/sum(exp(alpha))
counts = numeric(15)
for(i in 1:Nstint){
  these = sample(1:15,size=5,prob = probs)
  players = numeric(15)
  players[these] =1
  counts = counts + players
}

p = counts/Nstint
matrix(round(p * 48),ncol=5,byrow = TRUE)

install.packages("BART")
library(BART)

train <- read.csv("bart_train.csv")
test <- read.csv("bart_test.csv")

X_train <- as.matrix(train[, -ncol(train)])
y_train <- train$Y

X_test <- as.matrix(test[, -ncol(test)])
y_test <- test$Y

bart_model <- wbart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test
)

bart_predictions <- bart_model$yhat.test.mean

bart_rmse <- sqrt(mean((y_test - bart_predictions)^2))

print(bart_rmse)