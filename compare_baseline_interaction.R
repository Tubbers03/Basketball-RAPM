# ============================================================
# BASELINE VS HIDDEN INTERACTION EXPERIMENT
# OLS vs RIDGE vs BART
# ============================================================

rm(list = ls())

# ------------------------------------------------------------
# LOAD LIBRARIES
# ------------------------------------------------------------

library(glmnet)
library(BART)
library(MASS)

# ------------------------------------------------------------
# SET WORKING DIRECTORY
# ------------------------------------------------------------

setwd("/Users/aricalley/Documents/Research Project")


# ============================================================
# FUNCTION TO RUN ALL THREE MODELS
# ============================================================

run_models <- function(train_file, test_file, experiment_name) {

  cat("\n\n")
  cat("========================================\n")
  cat(experiment_name, "\n")
  cat("========================================\n")


  # ----------------------------------------------------------
  # LOAD DATA
  # ----------------------------------------------------------

  train <- read.csv(
    train_file,
    check.names = TRUE
  )

  test <- read.csv(
    test_file,
    check.names = TRUE
  )


  # ----------------------------------------------------------
  # CREATE X AND Y
  # ----------------------------------------------------------

  X_train <- train[, names(train) != "Y"]
  y_train <- train$Y

  X_test <- test[, names(test) != "Y"]
  y_test <- test$Y

  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)


  cat("\nTraining observations:", nrow(X_train), "\n")
  cat("Training predictors:", ncol(X_train), "\n")
  cat("Test observations:", nrow(X_test), "\n")
  cat("Test predictors:", ncol(X_test), "\n")


  # ==========================================================
  # BART
  # ==========================================================

  cat("\n----------------------------------------\n")
  cat("BART\n")
  cat("----------------------------------------\n")

  set.seed(123)

  bart_model <- wbart(
    x.train = X_train,
    y.train = y_train,
    x.test = X_test,
    ntree = 200,
    nskip = 100,
    ndpost = 1000
  )

  bart_predictions <- bart_model$yhat.test.mean

  bart_rmse <- sqrt(
    mean(
      (y_test - bart_predictions)^2
    )
  )

  cat(
    "BART Test RMSE:",
    bart_rmse,
    "\n"
  )


  # ==========================================================
  # OLS
  # ==========================================================

  cat("\n----------------------------------------\n")
  cat("OLS\n")
  cat("----------------------------------------\n")

  ols_beta <- MASS::ginv(
    X_train
  ) %*%
    y_train

  ols_beta <- as.numeric(
    ols_beta
  )

  ols_predictions <-
    X_test %*%
    ols_beta

  ols_rmse <- sqrt(
    mean(
      (y_test - ols_predictions)^2
    )
  )

  cat(
    "OLS Test RMSE:",
    ols_rmse,
    "\n"
  )


  # ==========================================================
  # RIDGE
  # ==========================================================

  cat("\n----------------------------------------\n")
  cat("RIDGE\n")
  cat("----------------------------------------\n")

  set.seed(123)

  ridge_model <- cv.glmnet(
    x = X_train,
    y = y_train,
    alpha = 0,
    nfolds = 10,
    standardize = FALSE
  )

  ridge_predictions <- predict(
    ridge_model,
    newx = X_test,
    s = "lambda.min"
  )

  ridge_predictions <- as.numeric(
    ridge_predictions
  )

  ridge_rmse <- sqrt(
    mean(
      (y_test - ridge_predictions)^2
    )
  )

  cat(
    "Best Ridge lambda:",
    ridge_model$lambda.min,
    "\n"
  )

  cat(
    "Ridge Test RMSE:",
    ridge_rmse,
    "\n"
  )


  # ==========================================================
  # RETURN RESULTS
  # ==========================================================

  return(
    list(
      OLS_RMSE = ols_rmse,
      Ridge_RMSE = ridge_rmse,
      BART_RMSE = bart_rmse,

      BART_Model = bart_model,

      OLS_Beta = ols_beta,

      Ridge_Model = ridge_model,

      X_Train = X_train
    )
  )
}


# ============================================================
# RUN BASELINE
# ============================================================

baseline_results <- run_models(
  train_file = "bart_train.csv",
  test_file = "bart_test.csv",
  experiment_name = "BASELINE EXPERIMENT"
)


# ============================================================
# RUN HIDDEN INTERACTION EXPERIMENT
# ============================================================

interaction_results <- run_models(
  train_file = "interaction_train.csv",
  test_file = "interaction_test.csv",
  experiment_name = "HIDDEN INTERACTION EXPERIMENT"
)


# ============================================================
# EXTRACT RMSE RESULTS
# ============================================================

baseline_ols_rmse <-
  baseline_results$OLS_RMSE

baseline_ridge_rmse <-
  baseline_results$Ridge_RMSE

baseline_bart_rmse <-
  baseline_results$BART_RMSE


interaction_ols_rmse <-
  interaction_results$OLS_RMSE

interaction_ridge_rmse <-
  interaction_results$Ridge_RMSE

interaction_bart_rmse <-
  interaction_results$BART_RMSE


# ============================================================
# COMPARE RESULTS
# ============================================================

comparison_results <- data.frame(

  Model = c(
    "OLS",
    "Ridge",
    "BART"
  ),

  Baseline_RMSE = c(
    baseline_ols_rmse,
    baseline_ridge_rmse,
    baseline_bart_rmse
  ),

  Interaction_RMSE = c(
    interaction_ols_rmse,
    interaction_ridge_rmse,
    interaction_bart_rmse
  )
)


# ------------------------------------------------------------
# RMSE CHANGE
# ------------------------------------------------------------

comparison_results$RMSE_Change <-
  comparison_results$Interaction_RMSE -
  comparison_results$Baseline_RMSE


# ------------------------------------------------------------
# PERCENT CHANGE
# ------------------------------------------------------------

comparison_results$Percent_Change <-
  (
    comparison_results$Interaction_RMSE -
      comparison_results$Baseline_RMSE
  ) /
  comparison_results$Baseline_RMSE *
  100


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

cat("\n\n")
cat("========================================\n")
cat("BASELINE VS HIDDEN INTERACTION RESULTS\n")
cat("========================================\n")

print(
  comparison_results,
  row.names = FALSE
)


# ============================================================
# SAVE COMPARISON
# ============================================================

write.csv(
  comparison_results,
  "baseline_vs_interaction_comparison.csv",
  row.names = FALSE
)


# ============================================================
# DONE
# ============================================================

cat("\n========================================\n")
cat("COMPARISON COMPLETE\n")
cat("========================================\n")