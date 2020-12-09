# **Final Report**

# Abstract
Due to the recent police activity and call for police reform over the past few months, we want to predict the outcome of a police stop based on a variety of attributes. We thought of the idea based on recent events around police reform and wanted to implement some way to see what happens and what will happen at a police stop. We want to look into what factors are reported for a police stop and see what will be the most important when predicting the stop_outcome.

# Experiments/Analysis
We decided to run the dataset on four different models, Random Forest, Decision Tree, K Nearest Neighbors, and Logistic Regression, to see how the models will perform. We also wanted to find some other details like which column is most important to predict the stop_outcome and if different depths on Decision Tree would make a difference.

## Random Forest


## Decision Tree
* Apply Decision Tree classifier algorithm on the dataset and compare how different approaches in implementing the algorithm impacts the accuracy
* The first approach is to find out the best parameters using the grid search
* In the second approach we try to remove individually one column at a time and try to find out 
  the accuracy respectively. This way we can find out which column is affecting the outcome much or
  indicating the importance of each column
* The third approach is to find accuracy by varying the depth. The Depth parameter for the Decision classifier
  is varied. For various depths the accuracy is calculated. The increase in the maximum depth causes the algorithm to overifit, hence higher depths are not preferred

## K Nearest Neighbors
* Apply KNN algorithm on the dataset and compare how different approaches in implementing the algorithm impacts the accuracy
* The first approach is to apply KNN on the entire dataset by selecting n number of neighbours
* The second approach is to reduce the dataset into n number of features and then apply KNN on the dataset
* The third approach is to find accuracy by using cross_val_score with 5 folds on KNN
* Finally, drop one column at a time and find the importance of each column in the dataset

## Logistic Regression


# Comparisons


# Conclusion