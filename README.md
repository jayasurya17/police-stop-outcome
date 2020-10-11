# Police Stop Outcome

**Team members**
|Name   | GitHub username |
|--|--|
| Akash Aggarwal | Akash15o3 |
| Anthony Minaise | anthonyminaise|
| Jayasurya Pinaki | jayasurya17 |
| Sri Hari Duvvuri | duvvurisrihari |

**Data Source**
We plan on using the Stanford Open Policing Project dataset from Kaggle. [https://www.kaggle.com/faressayah/stanford-open-policing-project](https://www.kaggle.com/faressayah/stanford-open-policing-project)

**Description**
Due to the recent police activity and call for police reform over the past few months, we want to predict the outcome of a police stop based on a variety of attributes. We thought of the idea based on recent events around police reform and wanted to implement some way to see what happens and what will happen at a police stop. We want to look into what factors are reported for a police stop and see what will be the most important when predicting the outcome.

**Potential Methods**
For preprocessing, we were thinking about using One Hot encoding (using pandas get_dummies), perform some test training split on the dataset, and some dimensionality reduction after running on the training set. We were thinking about using decision trees (like random forest), neural network, Naive Bayes, chi-squared, or even k-nearest neighbors. We want to use at least 2-3 algorithms to see which one would give the best accuracies and report the outcome.

**Metrics  to measure success**
We’ll measure success based on the accuracy of the test dataset portion of the overall dataset after running the algorithms on the training set. We will divide our dataset into a 70-30 split for training and testing.
