###Credit Card Fraud

#Dataset:
credit_data <- read_csv("~/Downloads/creditcard.csv")

#Quick path change
#Logistic
data <- read_csv("~/Downloads/creditcard.csv")
#Random Forest
data1 <- read_csv("~/Downloads/creditcard.csv")
#XGBoost
data2 <- read_csv("~/Downloads/creditcard.csv")
#Support Vector Machine
data3 <- read_csv("~/Downloads/creditcard.csv")
#K-means clustering
data4 <- read_csv("~/Downloads/creditcard.csv")
#Isolation forest
data5 <- read_csv("~/Downloads/creditcard.csv")

#Note: datasets are specialized to the model but 'train_data', 'test_data', and
#confusion matrix dfs are named the same
#For optimal use, I recommend clearing the environment between models


#Libraries
library(tidyverse)
library(caret)
library(xgboost)
library(dplyr)
library(randomForest)
library(e1071)
library(ggplot2)
library(reshape2)
library(isotree)
library(dbscan)

###Supervised:

##Logistic Regression
#Using 'data' for dataset

data <- read_csv("~/Downloads/creditcard.csv")

#Exploratory
str(data)
summary(data)

#target variable as factor
data$Class <- as.factor(data$Class)

#Test/Train split
set.seed(123)
split <- sample(1:nrow(data), size = 0.7 * nrow(data))
train_data <- data[split, ]
test_data <- data[-split, ]

#Logistic regression model
logit_model <- glm(Class ~ Time + V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 +
                     V11 + V12 + V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20+
                     V21 + V22 + V23 + V24 + V25 + V26 + V27 + V28 + Amount, 
                   data = train_data, 
                   family = binomial)

summary(logit_model)

#Predict on the test set
predictions <- predict(logit_model, newdata = test_data, type = "response")

#Convert probabilities to binary predictions (Using 0.5 threshold)
test_data$predicted_fraud <- ifelse(predictions > 0.5, 1, 0)

#Evaluate model performance
conf_matrix <- table(test_data$Class, test_data$predicted_fraud)
print(conf_matrix)

#Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")

#Visualizing the confusion matrix
conf_table <- as.table(conf_matrix)

#Convert to df for ggplot
conf_df <- as.data.frame(conf_table)
names(conf_df) <- c("Actual", "Predicted", "Freq")

#Heatmap
ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  labs(title = "Logistic Regression Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()


#-----------------------------------------------------------------------------
##Random Forest
#Using 'data1' for dataset

data1 <- read_csv("~/Downloads/creditcard.csv")

data1$Class <- as.factor(data1$Class)

#Train/Test split
set.seed(123)
train_index <- createDataPartition(data1$Class, p = 0.7, list = FALSE)
train_data <- data1[train_index, ]
test_data <- data1[-train_index, ]

#Random Forest model
rf_model <- randomForest(Class ~ Time + V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 +
                           V11 + V12 + V13 + V14 + V15 + V16 + V17 + V18 + V19 + V20+
                           V21 + V22 + V23 + V24 + V25 + V26 + V27 + V28 + Amount, 
                         data = train_data, 
                         ntree = 500,  #Number of trees
                         mtry = 2,     #Variables tried at each split
                         importance = TRUE)

#runtime ~5min on 4 cores
print(rf_model)

#Variable importance plot
varImpPlot(rf_model, main = "Variable Importance")

#Predict on test set
rf_predictions <- predict(rf_model, test_data)


#Confusion matrix
conf_matrix <- confusionMatrix(rf_predictions, test_data$Class)
print(conf_matrix)

conf_df <- as.data.frame(as.table(conf_matrix$table))
names(conf_df) <- c("Actual", "Predicted", "Freq")

ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  labs(title = "Random Forest Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

#----------------------------------------------------------------------------

#XGBoost
#Using 'data2' for dataset

data2 <- read_csv("~/Downloads/creditcard.csv")

#Train/Test split
set.seed(123)
train_index <- createDataPartition(data2$Class, p = 0.7, list = FALSE)
train_data <- data2[train_index, ]
test_data <- data2[-train_index, ]

#Ensure numeric types only
sapply(train_data[, -which(names(train_data) == "Class")], class)

#Ensuring target variable is 0 or 1
train_data$Class <- as.numeric(as.factor(train_data$Class)) - 1
test_data$Class <- as.numeric(as.factor(test_data$Class)) - 1


#Prepare data for model
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "Class")]),
                            label = train_data$Class)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "Class")]),
                           label = test_data$Class)

#Train the XGBoost model
xgb_model <- xgboost(data = train_matrix, 
                     max_depth = 6,
                     eta = 0.3,
                     nrounds = 100,
                     objective = "binary:logistic",
                     eval_metric = "logloss",
                     verbose = 0)


print(xgb_model)

#Feature importance
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance, main = "Feature Importance")

#Predict on test set
xgb_predictions <- predict(xgb_model, test_matrix)
xgb_predicted_class <- ifelse(xgb_predictions > 0.5, 1, 0)

#Confusion matrix
conf_matrix <- confusionMatrix(factor(xgb_predicted_class), factor(test_data$Class))
print(conf_matrix)

conf_df <- as.data.frame(as.table(conf_matrix$table))
names(conf_df) <- c("Actual", "Predicted", "Freq")

ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  labs(title = "XGBoost Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

#----------------------------------------------------------------------------

##Support Vector machines
#Using 'data3' for dataset

data3 <- read_csv("~/Downloads/creditcard.csv")

#Train/Test split
set.seed(123)
train_index <- createDataPartition(data3$Class, p = 0.7, list = FALSE)
train_data <- data3[train_index, ]
test_data <- data3[-train_index, ]

#Convert target variable to factor
train_data$Class <- as.factor(train_data$Class)
test_data$Class <- as.factor(test_data$Class)

#Train SVM model
svm_model <- svm(
  Class ~ .,
  data = train_data,
  kernel = "radial",
  cost = 1,
  gamma = 0.1
)


summary(svm_model)


#Predict on test data
svm_predictions <- predict(svm_model, test_data)

#Confusion matrix
conf_matrix <- table(Predicted = svm_predictions, Actual = test_data$Class)
print(conf_matrix)

#Accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")

#Precision, Recall, F1-score
confusion_stats <- confusionMatrix(svm_predictions, test_data$Class)
print(confusion_stats)


#Confusion Martix
conf_matrix <- table(Predicted = svm_predictions, Actual = test_data$Class)

conf_matrix_df <- as.data.frame(as.table(conf_matrix))
names(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "SVM Confusion Matrix",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal()

#Tune SVM model
#Crashed program after 4 hours of running
tuned_model <- tune(
  svm,
  Class ~ .,
  data = train_data,
  kernel = "radial",
  ranges = list(cost = 10^(-1:2), gamma = c(0.01, 0.1, 1))
)

#Best model parameters
best_model <- tuned_model$best.model
summary(best_model)

#Predictions using the best model
best_predictions <- predict(best_model, test_data)
best_conf_matrix <- table(Predicted = best_predictions, Actual = test_data$Class)
print(best_conf_matrix)


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
##Unsupervised:

#K-means clustering
#Using 'data4' for dataset

data4 <- read_csv("~/Downloads/creditcard.csv")

#Train/Test split
set.seed(123)
train_index <- createDataPartition(data4$Class, p = 0.7, list = FALSE)
train_data <- data4[train_index, ]
test_data <- data4[-train_index, ]

#Data prep
clustering_data <- scale(as.matrix(train_data[, -which(names(train_data) == "Class")]))



#Compute the total WSS for different k values
wss <- sapply(1:10, function(k) {
  kmeans(clustering_data, centers = k, nstart = 10)$tot.withinss
})

#Plot the Elbow Curve
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Sum of Squares (WSS)",
     main = "Elbow Method for Determining Optimal k")

#Apply K-means clustering
set.seed(123)
kmeans_model <- kmeans(clustering_data, centers = 5, nstart = 10)

#Add cluster assignments to the dataset
train_data$cluster <- kmeans_model$cluster

#Check the number of data points in each cluster
table(train_data$cluster)

#View cluster centroids
print(kmeans_model$centers)

#Perform PCA for visualization
pca_result <- prcomp(clustering_data, center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca_result$x)
pca_data$cluster <- as.factor(train_data$cluster)

#Plot the clusters

ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(
    title = "K-means Clustering Visualization",
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  theme_minimal()

table(Predicted_Cluster = train_data$cluster, Actual_Fraud = train_data$Class)

#Confusion matrix

conf_matrix <- table(Cluster = train_data$cluster, Actual = train_data$Class)

conf_matrix_df <- as.data.frame(as.table(conf_matrix))
names(conf_matrix_df) <- c("Cluster", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Cluster, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Confusion Matrix Heatmap - K-means Clustering",
    x = "Actual Fraud Labels",
    y = "Predicted Clusters",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(angle = 0)
  )

#-------------------------------------------------------------------------------

#Isolation forest
#Using 'data5' for dataset

data5 <- read_csv("~/Downloads/creditcard.csv")

#Train/Test split
set.seed(123)
train_index <- createDataPartition(data5$Class, p = 0.7, list = FALSE)
train_data <- data5[train_index, ]
test_data <- data5[-train_index, ]

#Data prep
train_data_matrix <- as.matrix(train_data[, -which(names(train_data) == "Class")])
test_data_matrix <- as.matrix(test_data[, -which(names(test_data) == "Class")])

#Train Isolation Forest model
iso_forest <- isolation.forest(
  train_data_matrix,
  ntrees = 100,      # Number of trees
  sample_size = 256, # Sample size for each tree
  nthreads = 1       # Number of threads for parallel processing
)

print(iso_forest)

#Predict anomaly scores for the test dataset
anomaly_scores <- predict(iso_forest, test_data_matrix, type = "score")

#Add anomaly scores to the test data
test_data$anomaly_score <- anomaly_scores

#Define a threshold (95th percentile of scores)
threshold <- quantile(anomaly_scores, 0.95)

#Flag anomalies
test_data$anomaly <- ifelse(test_data$anomaly_score > threshold, 1, 0)

table(test_data$anomaly)

#Confusion matrix

conf_matrix <- table(Predicted = test_data$anomaly, Actual = test_data$Class)
print(conf_matrix)

ggplot(test_data, aes(x = anomaly_score, fill = as.factor(Class))) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.7) +
  labs(
    title = "Anomaly Score Distribution",
    x = "Anomaly Score",
    y = "Count",
    fill = "Class"
  ) +
  theme_minimal()

#Confusion matrix as heatmap
conf_matrix <- table(Predicted = test_data$anomaly, Actual = test_data$Class)


conf_matrix_df <- as.data.frame(as.table(conf_matrix))
names(conf_matrix_df) <- c("Predicted", "Actual", "Count")


ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Confusion Matrix Heatmap - Isolation Forest",
    x = "Actual Labels",
    y = "Predicted Labels",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(angle = 0)
  )

#----------------------------------------------------------------------------
#Local Outlier Factor
#Using 'data6' for dataset
#K=100 has deprecated value of k -dont use this model

data6 <- read_csv("~/Downloads/creditcard.csv")

#Train/Test split
set.seed(123)
train_index <- createDataPartition(data6$Class, p = 0.7, list = FALSE)
train_data <- data6[train_index, ]
test_data <- data6[-train_index, ]

#Data prep
lof_data <- as.matrix(scale(train_data[, -which(names(train_data) == "Class")]))

#k=100 k being nearest neighbors
#Apply LOF
set.seed(123) # For reproducibility
lof_scores <- lof(lof_data, k = 100)
#Runtime ~25 minutes on 4 cores
#K is deprecated at k=100, see below for larger k value

#Add LOF scores to the dataset
train_data$lof_score <- lof_scores

#Set a threshold for anomalies
threshold <- quantile(train_data$lof_score, 0.95) # Top 5% as anomalies

#Classify observations as normal (0) or anomaly (1)
train_data$anomaly <- ifelse(train_data$lof_score > threshold, 1, 0)

#Confusion matrix
conf_matrix <- table(Predicted = train_data$anomaly, Actual = train_data$Class)
print(conf_matrix)

conf_matrix_df <- as.data.frame(as.table(conf_matrix))
names(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Confusion Matrix Heatmap - Local Outlier Factor",
    x = "Actual Labels",
    y = "Predicted Labels",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(angle = 0)
  )

#Visualize LOF Scores
ggplot(train_data, aes(x = lof_score, fill = as.factor(anomaly))) +
  geom_histogram(binwidth = 0.1, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("blue", "red"), labels = c("Normal", "Anomaly")) +
  labs(
    title = "LOF Score Distribution",
    x = "LOF Score",
    y = "Count",
    fill = "Class"
  ) +
  theme_minimal()

#---------------------------------
#k=1000 , k being nearest neighbors
#Apply LOF
set.seed(123) # For reproducibility
lof_scores <- lof(lof_data, k = 1000)
#Runtime ~20 minutes on 4 cores
#K is deprecated at k=65

#Add LOF scores to the dataset
train_data$lof_score <- lof_scores

#Set a threshold for anomalies
threshold <- quantile(train_data$lof_score, 0.95) # Top 5% as anomalies

#Classify observations as normal (0) or anomaly (1)
train_data$anomaly <- ifelse(train_data$lof_score > threshold, 1, 0)

#Confusion matrix
conf_matrix <- table(Predicted = train_data$anomaly, Actual = train_data$fraud)
print(conf_matrix)

conf_matrix_df <- as.data.frame(as.table(conf_matrix))
names(conf_matrix_df) <- c("Predicted", "Actual", "Count")

ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Confusion Matrix Heatmap - Local Outlier Factor",
    x = "Actual Labels",
    y = "Predicted Labels",
    fill = "Count"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(angle = 0)
  )

#Visualize LOF Scores
ggplot(train_data, aes(x = lof_score, fill = as.factor(anomaly))) +
  geom_histogram(binwidth = 0.1, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("blue", "red"), labels = c("Normal", "Anomaly")) +
  labs(
    title = "LOF Score Distribution",
    x = "LOF Score",
    y = "Count",
    fill = "Class"
  ) +
  theme_minimal()

#--------------------------------END---------------------------------------------
