# Import required libraries
library(readr)      # For reading CSV files
library(dplyr)      # For data manipulation
library(glue)       # For dynamic string interpolation
library(yardstick)  # For model performance evaluation (accuracy, recall, F1-score)
library(caret)      # For machine learning and model training
library(boot)       # For cross-validation
library(pROC)       # For ROC curve and AUC calculation

# Read the dataset
cars <- read_csv('car_insurance.csv', show_col_types = FALSE)

# View data structure
str(cars)

# Check for missing values in each column
colSums(is.na(cars))

# Summary of key numeric variables (credit score and annual mileage)
summary(cars$credit_score)
summary(cars$annual_mileage)

# Fill missing values with the mean of respective columns
cars$credit_score[is.na(cars$credit_score)] <- mean(cars$credit_score, na.rm = TRUE)
cars$annual_mileage[is.na(cars$annual_mileage)] <- mean(cars$annual_mileage, na.rm = TRUE)

# Verify if all missing values have been handled
colSums(is.na(cars))

# Create a dataframe to store feature names
features_df <- data.frame(features = c(names(subset(cars, select = -c(id, outcome)))))

# Initialize an empty vector to store accuracies for univariate analysis
accuracies <- c()

# UNIVARIATE ANALYSIS
# Loop through each feature to evaluate its performance individually
for (col in features_df$features) {
  # Create a logistic regression model using each feature
  model <- glm(glue('outcome ~ {col}'), data = cars, family = 'binomial')
  
  # Get the predicted values from the model
  predictions <- round(fitted(model))
  
  # Calculate accuracy
  accuracy <- length(which(predictions == cars$outcome)) / length(cars$outcome)
  
  # Store accuracy in the dataframe
  features_df[which(features_df$feature == col), "accuracy"] <- accuracy
}

# Find the feature with the highest accuracy
best_feature <- features_df$features[which.max(features_df$accuracy)]
best_accuracy <- max(features_df$accuracy)

# Create a dataframe with the best feature and its accuracy
best_feature_df <- data.frame(best_feature, best_accuracy)

# Print the best feature and its accuracy
print(best_feature_df)

# MULTIVARIATE ANALYSIS

# Encode categorical features already processed (One-hot encoding)
# gender, vehicle_ownership, married, children, vehicle_type, duis, outcome <- one-hot encoded

# Splitting the data into training and test sets
set.seed(123)
train_index <- sample(seq_len(nrow(cars)), size = 0.8 * nrow(cars))
train_set <- cars[train_index, ]
test_set <- cars[-train_index, ]

# Train a logistic regression model using all features
train_model <- glm(outcome ~ ., data = train_set, family = 'binomial')

# Predict the outcomes on the test set
predictions_test <- round(predict(train_model, newdata = test_set, type = 'response'))

# Calculate accuracy on the test set
accuracy_test <- length(which(predictions_test == test_set$outcome)) / length(test_set$outcome)
print(paste("Test Set Accuracy:", round(accuracy_test, 4)))

# Calculate error rate on the test set
error_test <- 1 - accuracy_test
print(paste("Test Set Error Rate:", round(error_test, 4)))

# K-fold cross-validation on the training set
cv_error_train <- cv.glm(train_set, train_model, K=10)
print(paste("Cross-validation Error (10-fold):", round(cv_error_train$delta[1], 4)))

# Backward elimination to optimize the model and reduce features
optimized_model <- step(train_model, direction = 'backward')
summary(optimized_model)

# Remove non-significant features and retrain the model
optimized_model1 <- glm(outcome ~ gender + driving_experience + vehicle_ownership + 
                          vehicle_year + married + children + postal_code + 
                          annual_mileage + past_accidents, family = "binomial", data = train_set)
summary(optimized_model1)

# Performance evaluation of the optimized model on the test set
predictions_test <- round(predict(optimized_model1, newdata = test_set, type = "response"))
accuracy_test <- length(which(predictions_test == test_set$outcome)) / length(test_set$outcome)
print(paste("Test Set Accuracy (Optimized Model):", round(accuracy_test, 4)))

# Confusion matrix for the test set
conf_matrix <- table(Predicted = predictions_test, Actual = test_set$outcome)
print(conf_matrix)

# ROC curve and AUC calculation for the optimized model
predictions_prob <- predict(optimized_model1, newdata = test_set, type = 'response')
roc_curve <- roc(test_set$outcome, predictions_prob)
plot(roc_curve)
auc_value <- auc(roc_curve)
print(paste("Area Under the Curve (AUC):", round(auc_value, 4)))
