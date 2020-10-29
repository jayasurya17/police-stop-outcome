# Police Stop Outcome

## Team members
|Name   | GitHub username |
|--|--|
| Akash Aggarwal | Akash15o3 |
| Anthony Minaise | anthonyminaise|
| Jayasurya Pinaki | jayasurya17 |
| Sri Hari Duvvuri | duvvurisrihari |

## Data Source
We plan on using the Stanford Open Policing Project dataset from Kaggle. [https://www.kaggle.com/faressayah/stanford-open-policing-project](https://www.kaggle.com/faressayah/stanford-open-policing-project)

## Description
Due to the recent police activity and call for police reform over the past few months, we want to predict the outcome of a police stop based on a variety of attributes. We thought of the idea based on recent events around police reform and wanted to implement some way to see what happens and what will happen at a police stop. We want to look into what factors are reported for a police stop and see what will be the most important when predicting the stop_outcome.
### Prediction
The stop_outcome column is what happens after an officer concludes a stop on a person, whether its for a citation, arrest, warning, or other possible outcomes. After performing a test train split on the dataset, we want to predict the stop_outcome using a few different machine learning algorithms to determine which algoirthm predicts the best for the given dataset.

## Potential Methods
For preprocessing, we were thinking about using One Hot encoding (using pandas get_dummies), perform some test training split on the dataset, and some dimensionality reduction after running on the training set. We were thinking about using decision trees (like random forest), Naive Bayes, chi-squared, or even k-nearest neighbors. We want to use at least 2-3 algorithms to see which one would give the best accuracies and report the outcome. We aren't too well versed in some of these algorithms, so we are not sure how the implementation will end up. We want to use this project as a chance to dive deeper into different algorithms and figure out how they work and how they are implemented after being preprocessed. 

## Metrics to Measure Success
We’ll measure success based on the accuracy of the test dataset portion of the overall dataset after running the algorithms on the training set. We will divide our dataset into a 70-30 split for training and testing. We hope to reach around 85-90% accuracy for the models.
### Outcome
At the end of the project, we want to be able to present statistics visually on the different algorithms on how they predicted versus what the actual stop_outcomes were. Based on the different columns like driver_race, stop_time, driver_age, search_conducted, violation, and gender, we want to use these as parameters to be passed into our model while also removing them from the main Dataframe and predict the stop_outcome. We hope that the end user, being some police officer or higher up, will be able to input parameters surrounding a stop into our model(s) and predict the stop outcome before it might happen.