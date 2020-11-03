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

## Need for the Project
Predicting a type of crime before it happens is an essential part of crime patrolling. Police officers make over 50,000 traffic arrests on an average day in the United States. This project requires attention to answer some questions that results into a stop by police. Some of these needs are :  

1. How often a driver is friskd?
2. How does drug activity change according to the time of the day?
3. Does more stops happen at night or the day?
4. Does gender play a factor resulting in a police stop?




### Prediction
The stop_outcome column is what happens after an officer concludes a stop on a person, whether its for a citation, arrest, warning, or other possible outcomes. After performing a test train split on the dataset, we want to predict the stop_outcome using a few different machine learning algorithms to determine which algoirthm predicts the best for the given dataset.



## Potential Methods

### Preprocessing 
For preprocessing, we were thinking about using One Hot encoding (using pandas get_dummies), perform some test training split on the dataset, and some dimensionality reduction after running on the training set. 

Ex: Change stop time from exact time stamps to time of the day (12AM-3AM, 3AM-6AM and so on)


### Methods Considered 
We were thinking about using decision trees, random forest or  k-nearest neighbors. We want to use at least 2-3 algorithms to see which one would give the best accuracies and report the outcome. We aren't too well versed in some of these algorithms, so we are not sure how the implementation will end up. We want to use this project as a chance to dive deeper into different algorithms and figure out how they work and how they are implemented after being preprocessed. 

- Decision Trees: It would be our first choice of metric and we will start classifying the data considering parameters like violation type, gender, time and so on thus reaching the stop outcome as our final branch. Henceforth, extracting a tree model depicting results.

- KNN: This will match data points given as input to the information available in the dataset. The prediction will be based on the closest match to a stop that happened in the past. 

- Random Forest: Multiple trees are grown instead of just one tree in decision tree algorithm. When a new object is given, the classification happens based on number of votes by the trees and it takes the average of the outputs in case of regression. This algorithm will also handle missing values and maintains accuracy for missing data. Additionally, it will not overfit the model as well. 


## Metrics to Measure Success
We’ll measure success based on the accuracy of the test dataset portion of the population dataset after running the algorithms on the training set. We will divide our dataset into a 70-30 split for training and testing. We hope to reach around 85-90% accuracy for the models. 

In the perspective of the project, Our aim is to assist researchers, journalists and policymakers in researching and improving police-public interactions.

## Outcome
At the end of the project, we want to be able to present statistics visually on the different algorithms on how they predicted versus what the actual stop_outcomes were. Based on the different columns like driver_race, stop_time, driver_age, search_conducted, violation, and gender, we want to use these as parameters to be passed into our model and predict the stop_outcome that can act as an additional assistance to the police force. 