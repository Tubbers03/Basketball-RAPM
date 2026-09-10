# ============================================================
# RAPM MODEL COMPARISON
# BART vs OLS vs RIDGE
# ============================================================

# ============================================================
# 1. INSTALL PACKAGES
# ============================================================
# Run these ONLY if you haven't installed the packages yet.
# You can comment them out after they are installed.

# install.packages("BART")
# install.packages("glmnet")


# ============================================================
# 2. LOAD PACKAGES
# ============================================================

library(glmnet)
library(BART)


# ============================================================
# 3. SET WORKING DIRECTORY
# ============================================================

setwd("/Users/aricalley/Documents/Research Project")

cat("\n========================================\n")
cat("WORKING DIRECTORY\n")
cat("========================================\n")

print(getwd())


# ============================================================
# 4. CHECK FILES
# ============================================================

cat("\n========================================\n")
cat("FILES IN PROJECT FOLDER\n")
cat("========================================\n")

print(list.files())


# ============================================================
# 5. LOAD TRAINING AND TEST DATA
# ============================================================

cat("\n========================================\n")
cat("LOADING DATA\n")
cat("========================================\n")

train <- read.csv("bart_train.csv")
test <- read.csv("bart_test.csv")

cat("Training rows:", nrow(train), "\n")
cat("Testing rows:", nrow(test), "\n")

cat("\nTraining columns:\n")
print(names(train))


# ============================================================
# 6. CREATE X AND Y
# ============================================================

# Y is the response variable.
# Everything except the last column is being treated as X.

X_train <- as.matrix(train[, -ncol(train)])
y_train <- train$Y

X_test <- as.matrix(test[, -ncol(test)])
y_test <- test$Y


cat("\n========================================\n")
cat("DATA DIMENSIONS\n")
cat("========================================\n")

cat("X_train:", nrow(X_train), "rows x",
    ncol(X_train), "columns\n")

cat("X_test:", nrow(X_test), "rows x",
    ncol(X_test), "columns\n")

cat("y_train length:", length(y_train), "\n")
cat("y_test length:", length(y_test), "\n")


# ============================================================
# 7. BART
# ============================================================

cat("\n========================================\n")
cat("RUNNING BART\n")
cat("========================================\n")

bart_model <- wbart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test
)

bart_predictions <- bart_model$yhat.test.mean

bart_rmse <- sqrt(
  mean((y_test - bart_predictions)^2)
)

cat("\nBART RMSE:\n")
print(bart_rmse)


# ========================================
# OLS
# ========================================

cat("\n========================================\n")
cat("RUNNING OLS\n")
cat("========================================\n")

# Check rank
qr_X <- qr(X_train)

cat("X_train rank:", qr_X$rank, "\n")
cat("X_train columns:", ncol(X_train), "\n")

# Generalized inverse OLS
ols_beta <- MASS::ginv(X_train) %*% y_train

# Predictions
ols_predictions <- X_test %*% ols_beta

# RMSE
ols_rmse <- sqrt(mean((y_test - ols_predictions)^2))

cat("\nOLS RMSE:\n")
print(ols_rmse)
# ============================================================
# 9. RIDGE
# ============================================================

cat("\n========================================\n")
cat("RUNNING RIDGE\n")
cat("========================================\n")

ridge_model <- glmnet(
  X_train,
  y_train,
  alpha = 0,
  intercept = FALSE
)

# Use lambda = 10
ridge_predictions <- predict(
  ridge_model,
  newx = X_test,
  s = 10
)

ridge_rmse <- sqrt(
  mean((y_test - ridge_predictions)^2)
)

cat("\nRidge RMSE:\n")
print(ridge_rmse)


# ============================================================
# 10. COMPARE ALL THREE MODELS
# ============================================================

cat("\n========================================\n")
cat("FINAL MODEL COMPARISON\n")
cat("========================================\n")

comparison <- data.frame(
  Model = c(
    "OLS",
    "Ridge",
    "BART"
  ),
  
  RMSE = c(
    ols_rmse,
    ridge_rmse,
    bart_rmse
  )
)

print(comparison)


# ============================================================
# 11. SORT FROM BEST TO WORST
# ============================================================

cat("\n========================================\n")
cat("MODELS SORTED BY RMSE\n")
cat("========================================\n")

comparison_sorted <- comparison[
  order(comparison$RMSE),
]

print(comparison_sorted)


# ============================================================
# 12. FIND BEST MODEL
# ============================================================

best_model <- comparison_sorted$Model[1]
best_rmse <- comparison_sorted$RMSE[1]

cat("\n========================================\n")
cat("BEST MODEL\n")
cat("========================================\n")

cat(
  "Best model:",
  best_model,
  "\n"
)

cat(
  "RMSE:",
  best_rmse,
  "\n"
)


# ============================================================
# 13. PRINT RMSE DIFFERENCES
# ============================================================

cat("\n========================================\n")
cat("RMSE DIFFERENCES\n")
cat("========================================\n")

cat(
  "OLS RMSE:",
  ols_rmse,
  "\n"
)

cat(
  "Ridge RMSE:",
  ridge_rmse,
  "\n"
)

cat(
  "BART RMSE:",
  bart_rmse,
  "\n"
)

cat(
  "\nOLS - Ridge:",
  ols_rmse - ridge_rmse,
  "\n"
)

cat(
  "OLS - BART:",
  ols_rmse - bart_rmse,
  "\n"
)

cat(
  "Ridge - BART:",
  ridge_rmse - bart_rmse,
  "\n"
)


# ============================================================
# 14. PERCENT IMPROVEMENT
# ============================================================

cat("\n========================================\n")
cat("PERCENT IMPROVEMENT\n")
cat("========================================\n")

# Positive number means the second model has lower RMSE.

ridge_improvement_over_ols <-
  (ols_rmse - ridge_rmse) / ols_rmse * 100

bart_improvement_over_ols <-
  (ols_rmse - bart_rmse) / ols_rmse * 100

bart_improvement_over_ridge <-
  (ridge_rmse - bart_rmse) / ridge_rmse * 100


cat(
  "Ridge improvement over OLS:",
  ridge_improvement_over_ols,
  "%\n"
)

cat(
  "BART improvement over OLS:",
  bart_improvement_over_ols,
  "%\n"
)

cat(
  "BART improvement over Ridge:",
  bart_improvement_over_ridge,
  "%\n"
)


# ============================================================
# 15. DONE
# ============================================================

cat("\n========================================\n")
cat("DONE!\n")
cat("========================================\n")