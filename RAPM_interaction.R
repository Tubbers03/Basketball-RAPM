# ============================================================
# RAPM INTERACTION MODEL COMPARISON
# ============================================================

# Clear workspace
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

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

train <- read.csv(
  "interaction_train.csv",
  check.names = TRUE
)

test <- read.csv(
  "interaction_test.csv",
  check.names = TRUE
)

# ------------------------------------------------------------
# SEPARATE X AND Y
# ------------------------------------------------------------

X_train <- train[, names(train) != "Y"]
y_train <- train$Y

X_test <- test[, names(test) != "Y"]
y_test <- test$Y

# Convert to matrices
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

# ------------------------------------------------------------
# CHECK DATA
# ------------------------------------------------------------

cat("\n========================================\n")
cat("DATA CHECK\n")
cat("========================================\n")

cat("Training observations:", nrow(X_train), "\n")
cat("Training predictors:", ncol(X_train), "\n")
cat("Test observations:", nrow(X_test), "\n")
cat("Test predictors:", ncol(X_test), "\n")

cat("\nTraining Y range:\n")
print(range(y_train))

cat("\nTest Y range:\n")
print(range(y_test))

# ------------------------------------------------------------
# BART
# ------------------------------------------------------------

cat("\n========================================\n")
cat("RUNNING BART\n")
cat("========================================\n")

bart_model <- wbart(
  x.train = X_train,
  y.train = y_train,
  x.test = X_test,
  ntree = 200,
  nskip = 100,
  ndpost = 1000
)

# ------------------------------------------------------------
# BART TEST RMSE
# ------------------------------------------------------------

bart_predictions <- bart_model$yhat.test.mean

bart_rmse <- sqrt(
  mean((y_test - bart_predictions)^2)
)

cat("\nBART Test RMSE:", bart_rmse, "\n")

# ------------------------------------------------------------
# BART VARIABLE USAGE
# ------------------------------------------------------------

cat("\n========================================\n")
cat("BART VARIABLE USAGE\n")
cat("========================================\n")

bart_usage <- colMeans(bart_model$varcount)

bart_usage_df <- data.frame(
  Player = names(bart_usage),
  Usage = as.numeric(bart_usage)
)

bart_usage_df <- bart_usage_df[
  order(-bart_usage_df$Usage),
]

print(
  head(bart_usage_df, 30)
)

# ------------------------------------------------------------
# TRUE INTERACTION PLAYERS
# ------------------------------------------------------------

interaction_players <- c(
  "Ashley.Wilson",
  "Brooklyn",
  "Anna",
  "Hay.Hay",
  "CalebB",
  "Julio.Rodriguez",
  "Deni.Avdijia",
  "Whooly",
  "Handsome.Jack",
  "Rudy.Gobert",
  "Ben.Rice",
  "Ben",
  "Dailey",
  "Mystery.Girl",
  "Darius.Garland",
  "Brook.Lopez",
  "Ralph",
  "Buddy",
  "Shaggy",
  "Bugs",
  "Eli",
  "Mylee",
  "Mario",
  "Yoshi",
  "Jaxsyn",
  "Demar.Defrozen",
  "Shohei.Ohtani",
  "Freddie.Freeman",
  "Carl",
  "Jimmy",
  "Sheen",
  "Cindy",
  "Nate",
  "Narwhal",
  "Defender",
  "Attacker",
  "Toby",
  "Aric",
  "Brooke",
  "Adalee"
)

# Only keep interaction players that actually appear
# in BART's variable-usage output
interaction_players_present <- interaction_players[
  interaction_players %in% names(bart_usage)
]

cat("\nInteraction players found in BART:\n")
print(interaction_players_present)

# ------------------------------------------------------------
# INTERACTION PLAYER USAGE
# ------------------------------------------------------------

interaction_usage <- bart_usage[
  names(bart_usage) %in% interaction_players
]

cat("\nBART usage for interaction players:\n")

interaction_usage_df <- data.frame(
  Player = names(interaction_usage),
  Usage = as.numeric(interaction_usage)
)

interaction_usage_df <- interaction_usage_df[
  order(-interaction_usage_df$Usage),
]

print(interaction_usage_df)

# ------------------------------------------------------------
# COMPARE INTERACTION VS NON-INTERACTION PLAYERS
# ------------------------------------------------------------

is_interaction <- names(bart_usage) %in% interaction_players

cat("\nAverage BART usage:\n")

cat(
  "Interaction players:",
  mean(bart_usage[is_interaction]),
  "\n"
)

cat(
  "Other players:",
  mean(bart_usage[!is_interaction]),
  "\n"
)

# ------------------------------------------------------------
# OLS
# ------------------------------------------------------------

cat("\n========================================\n")
cat("RUNNING OLS\n")
cat("========================================\n")

# X is rank deficient, so use generalized inverse
ols_beta <- MASS::ginv(X_train) %*% y_train

ols_beta <- as.numeric(ols_beta)

ols_predictions <- X_test %*% ols_beta

ols_rmse <- sqrt(
  mean((y_test - ols_predictions)^2)
)

cat("\nOLS Test RMSE:", ols_rmse, "\n")

# ------------------------------------------------------------
# RIDGE WITH CROSS VALIDATION
# ------------------------------------------------------------

cat("\n========================================\n")
cat("RUNNING RIDGE\n")
cat("========================================\n")

ridge_model <- cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 0,
  nfolds = 10,
  standardize = FALSE
)

ridge_beta <- as.numeric(
  coef(
    ridge_model,
    s = "lambda.min"
  )
)[-1]

ridge_predictions <- predict(
  ridge_model,
  newx = X_test,
  s = "lambda.min"
)

ridge_predictions <- as.numeric(ridge_predictions)

ridge_rmse <- sqrt(
  mean((y_test - ridge_predictions)^2)
)

cat("\nBest Ridge lambda:", ridge_model$lambda.min, "\n")
cat("Ridge Test RMSE:", ridge_rmse, "\n")

# ------------------------------------------------------------
# LOAD TRUE ALPHAS
# ------------------------------------------------------------

cat("\n========================================\n")
cat("LOADING TRUE ALPHAS\n")
cat("========================================\n")

true_alpha_df <- read.csv(
  "true_alphas.csv",
  check.names = TRUE
)

# Convert both sets of player names to the same format
true_alpha_names <- make.names(true_alpha_df$Player)

true_alpha <- true_alpha_df$True_Alpha[
  match(
    colnames(X_train),
    true_alpha_names
  )
]

# Check for missing matches
if (any(is.na(true_alpha))) {

  cat("\nWARNING: Some players did not match true alphas.\n")

  print(
    colnames(X_train)[is.na(true_alpha)]
  )

} else {

  cat("\nAll 150 players matched true alphas successfully.\n")

}

# ------------------------------------------------------------
# CENTER COEFFICIENTS
# ------------------------------------------------------------

ols_centered <- ols_beta - mean(ols_beta)

ridge_centered <- ridge_beta - mean(ridge_beta)

true_centered <- true_alpha - mean(true_alpha)

# ------------------------------------------------------------
# ALPHA RMSE
# ------------------------------------------------------------

ols_alpha_rmse <- sqrt(
  mean(
    (ols_centered - true_centered)^2,
    na.rm = TRUE
  )
)

ridge_alpha_rmse <- sqrt(
  mean(
    (ridge_centered - true_centered)^2,
    na.rm = TRUE
  )
)

cat("\n========================================\n")
cat("ALPHA RECOVERY\n")
cat("========================================\n")

cat(
  "OLS alpha RMSE:",
  ols_alpha_rmse,
  "\n"
)

cat(
  "Ridge alpha RMSE:",
  ridge_alpha_rmse,
  "\n"
)

# ------------------------------------------------------------
# COEFFICIENT TABLE
# ------------------------------------------------------------

results_df <- data.frame(
  Player = colnames(X_train),
  True_Alpha = true_alpha,
  OLS = ols_centered,
  Ridge = ridge_centered
)

results_df$OLS_Error <- abs(
  results_df$OLS - results_df$True_Alpha
)

results_df$Ridge_Error <- abs(
  results_df$Ridge - results_df$True_Alpha
)

# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

cat("\n========================================\n")
cat("COEFFICIENT RESULTS\n")
cat("========================================\n")

print(results_df)

# ------------------------------------------------------------
# OVERALL MODEL COMPARISON
# ------------------------------------------------------------

cat("\n========================================\n")
cat("MODEL COMPARISON\n")
cat("========================================\n")

comparison_df <- data.frame(
  Model = c(
    "OLS",
    "Ridge",
    "BART"
  ),
  Test_RMSE = c(
    ols_rmse,
    ridge_rmse,
    bart_rmse
  )
)

print(comparison_df)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

write.csv(
  results_df,
  "interaction_alpha_results.csv",
  row.names = FALSE
)

write.csv(
  comparison_df,
  "interaction_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  bart_usage_df,
  "interaction_bart_variable_usage.csv",
  row.names = FALSE
)

cat("\n========================================\n")
cat("DONE\n")
cat("========================================\n")