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

# ============================================================
# 7.5. BART VARIABLE USAGE
# ============================================================

cat("\n========================================\n")
cat("BART VARIABLE USAGE\n")
cat("========================================\n")


# ------------------------------------------------------------
# Calculate average number of times BART uses each variable
# ------------------------------------------------------------

var_counts <- colMeans(
  bart_model$varcount
)


# ------------------------------------------------------------
# Create variable usage table
# ------------------------------------------------------------

bart_variable_usage <- data.frame(
  Player = colnames(bart_model$varcount),
  Usage = var_counts
)


# ------------------------------------------------------------
# Sort from most used to least used
# ------------------------------------------------------------

bart_variable_usage <- bart_variable_usage[
  order(
    bart_variable_usage$Usage,
    decreasing = TRUE
  ),
]


cat("\nTOP 30 VARIABLES USED BY BART\n")
cat("----------------------------------------\n")

print(
  head(
    bart_variable_usage,
    30
  ),
  row.names = FALSE
)


# ============================================================
# TRUE INTERACTION PAIRS
# ============================================================

interaction_pairs <- list(
  c("Ashley.Wilson", "Brooklyn"),
  c("Anna", "Hay.Hay"),
  c("CalebB", "Julio.Rodriguez"),
  c("Deni.Avdija", "Whooly"),
  c("Handsome.Jack", "Rudy.Gobert"),
  c("Ben.Rice", "Ben"),
  c("Dailey", "Mystery.Girl"),
  c("Darius.Garland", "Brook.Lopez"),
  c("Ralph", "Buddy"),
  c("Shaggy", "Bugs"),
  c("Eli", "Mylee"),
  c("Mario", "Yoshi"),
  c("Jaxsyn", "Demar.Defrozen"),
  c("Shohei.Ohtani", "Freddie.Freeman"),
  c("Carl", "Jimmy"),
  c("Sheen", "Cindy"),
  c("Nate", "Narwhal"),
  c("Defender", "Attacker"),
  c("Toby", "Aric"),
  c("Brooke", "Adalee")
)


# ------------------------------------------------------------
# Get all players involved in interactions
# ------------------------------------------------------------

interaction_players <- unique(
  unlist(interaction_pairs)
)


# ------------------------------------------------------------
# Mark interaction players
# ------------------------------------------------------------

bart_variable_usage$True_Interaction_Player <-
  bart_variable_usage$Player %in% interaction_players


# ------------------------------------------------------------
# Extract interaction players
# ------------------------------------------------------------

interaction_usage <- bart_variable_usage[
  bart_variable_usage$True_Interaction_Player,
]


interaction_usage <- interaction_usage[
  order(
    interaction_usage$Usage,
    decreasing = TRUE
  ),
]


cat("\n========================================\n")
cat("BART USAGE OF TRUE INTERACTION PLAYERS\n")
cat("========================================\n")

print(
  interaction_usage,
  row.names = FALSE
)


# ============================================================
# AVERAGE USAGE COMPARISON
# ============================================================

interaction_mean_usage <- mean(
  interaction_usage$Usage
)


noninteraction_mean_usage <- mean(
  bart_variable_usage[
    !bart_variable_usage$True_Interaction_Player,
    "Usage"
  ]
)


cat("\n========================================\n")
cat("AVERAGE VARIABLE USAGE\n")
cat("========================================\n")

cat(
  "True interaction players:",
  interaction_mean_usage,
  "\n"
)

cat(
  "Other players:",
  noninteraction_mean_usage,
  "\n"
)


# ========================================
# 8. OLS
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

# ------------------------------------------------------------
# Use cross-validation to choose lambda
# ------------------------------------------------------------

set.seed(123)

ridge_cv <- cv.glmnet(
  X_train,
  y_train,
  alpha = 0,
  intercept = FALSE,
  nfolds = 10
)

# Lambda chosen by minimum cross-validation error
best_lambda <- ridge_cv$lambda.min

cat("\nBest lambda from cross-validation:\n")
print(best_lambda)


# ------------------------------------------------------------
# Fit Ridge using the selected lambda
# ------------------------------------------------------------

ridge_model <- glmnet(
  X_train,
  y_train,
  alpha = 0,
  intercept = FALSE,
  lambda = best_lambda
)


# ------------------------------------------------------------
# Make predictions on the test set
# ------------------------------------------------------------

ridge_predictions <- predict(
  ridge_model,
  newx = X_test,
  s = best_lambda
)


# ------------------------------------------------------------
# Calculate test RMSE
# ------------------------------------------------------------

ridge_rmse <- sqrt(
  mean((y_test - ridge_predictions)^2)
)

cat("\nRidge RMSE:\n")
print(ridge_rmse)


# ============================================================
# 9.5. TRUE ALPHA RECOVERY
# ============================================================

cat("\n========================================\n")
cat("TRUE ALPHA RECOVERY\n")
cat("========================================\n")


# ------------------------------------------------------------
# Load true alpha values
# ------------------------------------------------------------

true_alphas <- read.csv("true_alphas.csv")

cat("\nTrue alpha rows:", nrow(true_alphas), "\n")
cat("\nTrue alpha columns:\n")
print(names(true_alphas))


# ------------------------------------------------------------
# Extract OLS coefficients
# ------------------------------------------------------------

ols_beta_vector <- as.vector(ols_beta)


# ------------------------------------------------------------
# Extract Ridge coefficients
# ------------------------------------------------------------

ridge_beta_vector <- as.vector(coef(ridge_model))

# Remove intercept if glmnet included one
if (length(ridge_beta_vector) == ncol(X_train) + 1) {
  ridge_beta_vector <- ridge_beta_vector[-1]
}


# ------------------------------------------------------------
# Create player effect table
# ------------------------------------------------------------

player_effects <- data.frame(
  Player = colnames(X_train),
  True_Alpha = true_alphas$True_Alpha,
  OLS = ols_beta_vector,
  Ridge = ridge_beta_vector
)


# ------------------------------------------------------------
# Center all coefficients
# ------------------------------------------------------------

player_effects$True_Alpha_Centered <-
  player_effects$True_Alpha -
  mean(player_effects$True_Alpha)

player_effects$OLS_Centered <-
  player_effects$OLS -
  mean(player_effects$OLS)

player_effects$Ridge_Centered <-
  player_effects$Ridge -
  mean(player_effects$Ridge)


# ------------------------------------------------------------
# Calculate errors
# ------------------------------------------------------------

player_effects$OLS_Error <-
  player_effects$OLS_Centered -
  player_effects$True_Alpha_Centered

player_effects$Ridge_Error <-
  player_effects$Ridge_Centered -
  player_effects$True_Alpha_Centered


player_effects$OLS_Absolute_Error <-
  abs(player_effects$OLS_Error)

player_effects$Ridge_Absolute_Error <-
  abs(player_effects$Ridge_Error)


# ------------------------------------------------------------
# Calculate coefficient RMSE
# ------------------------------------------------------------

ols_alpha_rmse <- sqrt(
  mean(player_effects$OLS_Error^2)
)

ridge_alpha_rmse <- sqrt(
  mean(player_effects$Ridge_Error^2)
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

cat("\n========================================\n")
cat("PLAYER EFFECT RMSE\n")
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
# Compare alpha recovery
# ------------------------------------------------------------

cat("\n========================================\n")
cat("ALPHA RECOVERY COMPARISON\n")
cat("========================================\n")

alpha_comparison <- data.frame(
  Model = c(
    "OLS",
    "Ridge"
  ),
  
  Alpha_RMSE = c(
    ols_alpha_rmse,
    ridge_alpha_rmse
  )
)

print(alpha_comparison)


# ------------------------------------------------------------
# Show individual player estimates
# ------------------------------------------------------------

cat("\n========================================\n")
cat("PLAYER EFFECT ESTIMATES\n")
cat("========================================\n")

print(
  player_effects[
    c(
      "Player",
      "True_Alpha",
      "OLS",
      "Ridge",
      "OLS_Error",
      "Ridge_Error"
    )
  ],
  row.names = FALSE
)


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