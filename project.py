import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import knn_analysis as knn_analysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression    #Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import resample
import common_utils as common_utils
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1

# Read CSV and return dataframe
# 
# Parameters
# filename: Location/pathname of CSV that needs be returned as a dataframe 
# 
# Return: dataframe from CSV
def read_file(filename):
    return pd.read_csv(filename)


# Save dataframe as CSV
# 
# Parameters
# df: Dataframe that needs to be converted into CSV file
# filename: Location/pathname where dataframe needs to be saved as CSV
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)


# Plot a pivot graph by grouping columns of choice in a dataframe and save the image
# 
# Parameters
# df: Dataframe that needs to be used for plotting the graph
# columns: Columns upon which grouping needs to be done
# filename: Location/pathname where the resulting graph must be saved
def group_and_plot_pivot_graph(df, columns, filename):

    # Group by one or more columns of choice in the dataframe
    grouped_multiple = df.groupby(columns).size().reset_index()
    # Rename the grouped column with size of groups as count
    grouped_multiple.columns = columns + ['count']

    fig = plt.figure()
    grouped_multiple.pivot(columns[0], columns[1], 'count').plot(kind='bar')
    plt.ylabel('count')
    plt.legend(title=columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)


# Plot a pie chart based on a particular column of choice in a dataframe and save the image
# 
# Parameters
# df: Dataframe that needs to be used for plotting the graph
# columns: Columns upon which graph needs to be plotted
# filename: Location/pathname where the resulting graph must be saved
def plot_pie_chart(df, column, filename):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(ax=ax, kind='pie')
    plt.tight_layout()
    plt.savefig(filename)


# Plot a bar graph based on a particular column of choice in a dataframe and save the image
# 
# Parameters
# df: Dataframe that needs to be used for plotting the graph
# columns: Columns upon which graph needs to be plotted
# filename: Location/pathname where the resulting graph must be saved
def plot_bar_graph(df, column, filename):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(ax=ax, kind='bar')
    plt.xlabel(column)
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(filename)


# Plot a kde graph based on a particular column of choice in a dataframe and save the image
# 
# Parameters
# df: Dataframe that needs to be used for plotting the graph
# columns: Columns upon which graph needs to be plotted
# filename: Location/pathname where the resulting graph must be saved
def plot_kde_graph(df, column, filename):
    fig, ax = plt.subplots()
    df[column].plot(ax=ax, kind='kde')
    plt.xlabel(column)
    plt.ylabel('Probability Distribution')
    plt.tight_layout()
    plt.savefig(filename)


# Plot a histogram based on a particular column of choice in a dataframe and save the image
# 
# Parameters
# df: Dataframe that needs to be used for plotting the graph
# columns: Columns upon which graph needs to be plotted
# filename: Location/pathname where the resulting graph must be saved
def plot_hist_graph(df, column, filename):
    fig, ax = plt.subplots()
    df[column].plot(ax=ax, kind='hist')
    plt.xlabel(column)
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(filename)

# Plot a line chart and save the image
# 
# Parameters
# x: values that needs to be present in x axis as a list
# y: corresponding y values that needs to be plotted as a list
# x_label: label that needs to be marked on x axis as a string
# y_label: label that needs to be marked on y axis as a string
# title: title for the graph as a string
# filename: Location/pathname where the resulting graph must be saved
def plot_line_chart(x, y, x_label, y_label, title, filename):
	fig, ax = plt.subplots()
	plt.plot(y, '--', marker='o')
	plt.xticks(range(0, len(x)), x)
	ax.set_xticklabels(x, rotation=(25), fontsize=8, ha='right')
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(filename)


def transform_data_columns_1_4(dataframe):
    # The county name column can be dropped cause it is completely null
    dataframe = dataframe.drop(columns=['county_name'], axis=1, inplace=False)

    # The gender dataset contains nulls. Here those nulls are discarded cause
    # the dataset is large and the nulls consitute about 5%. Also replacing the
    # null data with frequent values might mislead the prediction
    dataframe = dataframe.dropna(subset=['driver_gender'], inplace=False)

    # Create a colum that gives only the year of stop, which can be used
    # visualization of other columns with year
    dataframe['stop_year'] = dataframe['stop_date'].apply(lambda x: int(x.split('-')[0]))

    return dataframe


# Looks for missing or incorrect values in the row and returns mean value if any errors were found. Returns current value otherwise 
# 
# Parameters
# row: Row in which check must be performed
# mean_Age: Mean age of the drivers in dataset that needs to be returned in case of an error in the row
# 
# Return: "Corrected" age of driver
def get_driver_age(row, mean_age):
    age = row["stop_year"] - row["driver_age_raw"]
    if (age < 16 or row["driver_age"] == np.nan or age != row["driver_age"]):
        return mean_age
    return row["drivers_age_new"]


def transform_data_columns_5_8(df):
    # To clean the driver's age column and treat null values, we can replace it with the average value
    # of age in column where there is an incorrect value, null value or age less than 16.
    df["drivers_age_new"] = df["driver_age"]

    # To clean the driver's race column and treat null values, we can replace it with the most frequent value
    # of race in column
    df["drivers_race"] = df["driver_race"].fillna(df["driver_race"].mode())
    df = df.drop(columns=["driver_race"], axis=1, inplace=False)

    # To clean the violation column and treat null values, we can replace it with the most frequent value
    # of violation in column
    df["violations_raw"] = df["violation_raw"].fillna(df["violation_raw"].mode())
    df = df.drop(columns=["violation_raw"], axis=1, inplace=False)

    # x=0
    # y=0
    # z=int(df["driver_age"].mean())

    # for i in df.index:
    # 	x= df["stop_year"][i]-df["driver_age_raw"][i]
    # 	if(x<16 or df["driver_age"][i]==np.nan or x!= df["driver_age"][i]):
    # 		y=y+1
    # 		df["drivers_age_new"][i]=z

    # Find mean age of all drivers in the dataframe 
    mean_age = int(df["driver_age"].mean())
    # "Corrected" age of all drivers
    df["drivers_age_new"] = df.apply(lambda x: get_driver_age(x, mean_age), axis=1)

    # There are 2 similar columns that is driver's year of birth and driver's age
    # which provide converging information, pre-pruning is performed and driver's year of birth is
    # dropped as driver's age is sufficient for analysis.
    # It also helps in removing unneccessary branches before performing classification
    df = df.drop(columns=["driver_age"], axis=1, inplace=False)
    df = df.drop(columns=['driver_age_raw'], axis=1, inplace=False)

    return df


def visualize_data_columns_1_4(dataframe):
    # The statistics of the first four coumns are as follows

    # Column 1 -> Stop Date - The year on which the person is stopped for each gender
    plot_bar_graph(dataframe, 'stop_year', 'priliminary_visualization/stop_year.png')
    group_and_plot_pivot_graph(dataframe, ['stop_year', 'driver_gender'],
                               'priliminary_visualization/gender_with_year.png')

    # Column 2 -> Stop Time - The hour on which the person is stopped for each gender
    plot_bar_graph(dataframe, 'stop_hour', 'priliminary_visualization/stop_hour.png')
    group_and_plot_pivot_graph(dataframe, ['stop_hour', 'driver_gender'],
                               'priliminary_visualization/gender_with_hour.png')

    # Column 3 -> Gender
    print(dataframe['driver_gender'].value_counts())
    plot_pie_chart(dataframe, 'driver_gender', 'priliminary_visualization/driver_gender.png')

    return dataframe


def visualize_data_columns_5_8(df):
    # The statistics of the 5-8 coumns are as follows

    # Column 6 -> Driver's Race - The race of the driver who was stopped
    # Plotting a Pie-Chart representing race of the driver
    plot_pie_chart(df, 'drivers_race', 'priliminary_visualization/drivers_race.png')

    # Column 7 -> Violation Type - The type of violation by the driver
    # Plotting a Bar-Graph representing type of violation by the driver
    plot_bar_graph(df, 'violations_raw', 'priliminary_visualization/violations_raw.png')

    # Column 8 -> Driver's Age - The age of the driver who was stopped
    # Plotting a histogram and KDE plot capturing age of the driver 
    plot_kde_graph(df, 'drivers_age_new', 'priliminary_visualization/drivers_age.png')
    plot_hist_graph(df, 'drivers_age_new', 'priliminary_visualization/drivers_hist_age.png')


# Since multiple type of searches can be done on a single stop. A combination of these searches will result into an score based on weights of each search type.
# If a particular search was conducted, then 2 ^ weight of the search type is added to the total score
# This is done for each search that was conducted
# 
# Parameters
# weights: weights of each search type
# search_type: comma seperated list of search types that were conducted
def calculate_search_score(weights, search_type):
    search = search_type.split(',')
    total_weight = 0
    for value in search:
        total_weight += pow(2, weights[value])
    return total_weight


# Analyze and apply transformations that are required on the following colums
# violation, search_conducted, search_type, stop_outcome
# 
# Parameters
# df: dataframe in which all transformations must be applied
# 
# Return: dataframe after transformation
def transform_data_columns_9_12(df):
    # Get the hour of day at which stop happened
    df['stop_hour'] = df['stop_time'].apply(lambda x: int(x.split(':')[0]))

    # Fill the empty values with string 'None' and calculate search score by converting search types into a numberic value
    df['search_type'].fillna('None', inplace=True)
    lst = df['search_type'].str.split(',').explode().unique().tolist()
    weights = {}
    for i in range(len(lst)):
        weights[lst[i]] = i
    df['search_score'] = df['search_type'].apply(lambda x: calculate_search_score(weights, x))

    # Drop rows where the outcome is not avaliable
    df = df[df.stop_outcome != 'N/D']

    return df


def transform_data_columns_13_15(dataframe):
    # Replace 1 and 2 in the stop_duration column with 0-15 Min
    dataframe['stop_duration'] = dataframe['stop_duration'].str.replace("1", '0-15 Min')
    dataframe['stop_duration'] = dataframe['stop_duration'].str.replace("2", '0-15 Min')

    # Fix the other values that were impacted
    dataframe['stop_duration'] = dataframe['stop_duration'].str.replace("0-0-15 Min5 Min", '0-15 Min')
    dataframe['stop_duration'] = dataframe['stop_duration'].str.replace("0-15 Min6-30 Min", '16-30 Min')

    return dataframe


# Visualize the following colums using suitable graphs 
# violation, search_conducted, search_type, stop_outcome
# 
# Parameters
# df: dataframe where visualizations must be applied
def visualize_data_columns_9_12(df):

    # Plot a pie chart showing the percentage of cars where a search was conducted
    plot_pie_chart(df, 'search_conducted', 'priliminary_visualization/search_conducted.png')

    # Plot a bar graph showing the number of times a particular search was done
    df1 = df[df.search_type != 'None']
    s = df1['search_type'].str.split(',', expand=True).stack()
    index = s.index.get_level_values(0)
    df1 = df1.loc[index].copy()
    df1['search_type'] = s.values
    plot_bar_graph(df1, 'search_type', 'priliminary_visualization/search_type.png')

    # Plot a pivot graph showing the number and type of violation that were done per hour during the entire day
    group_and_plot_pivot_graph(df, ['stop_hour', 'violation'], 'priliminary_visualization/violation.png')

    # Plot a pivot graph showing the number and type of outcome that occurs based on the violation
    group_and_plot_pivot_graph(df, ['violation', 'stop_outcome'], 'priliminary_visualization/stop_outcome.png')


def visualize_data_columns_13_15(dataframe):
    plot_pie_chart(dataframe, 'is_arrested', 'priliminary_visualization/is_arrested.png')
    plot_pie_chart(dataframe, 'stop_duration', 'priliminary_visualization/stop_duration.png')
    plot_pie_chart(dataframe, 'drugs_related_stop', 'priliminary_visualization/drugs_related_stop.png')


# Using the date in YY-MM-DD format, split the value into its corresponding date, month and year and return the row
# 
# Parameters
# row: row where the date must be split
#
# Return: row with seperate columns for date, month and year
def split_date(row):
    date = row['stop_date'].split('-')
    row['stop_date'] = date[2]
    row['stop_month'] = date[1]
    row['stop_year'] = date[0]
    return row


def general_preprocessing(dataframe):

    # split stop_date into its corresponding date, month and year
    new = dataframe["stop_date"].str.split("-", n=2, expand=True)
    dataframe['stop_date'] = new[2]
    dataframe['stop_month'] = new[1]
    dataframe['stop_year'] = new[0]

    # Mark all the drivers based on the age group they fall into
    dataframe['drivers_age_bucket'] = dataframe['drivers_age_new'].apply(lambda x: x // 5)

    # Remove unwanted columns
    del dataframe['search_conducted']
    del dataframe['search_type']
    del dataframe['violation']
    del dataframe['stop_time']
    del dataframe['drivers_age_new']

    # Reindex columns in the dataframe
    column_names = ['stop_year', 'stop_month', 'stop_date', 'stop_hour', 'driver_gender', 'drivers_age_bucket',
                    'drivers_race', 'stop_duration', 'is_arrested', 'drugs_related_stop', 'violations_raw',
                    'search_score', 'stop_outcome']
    dataframe = dataframe.reindex(columns=column_names)

    dataframe.dropna()
    print("Processed dataframe")
    print(dataframe)

    return dataframe


# Visualize the Confusion Matrix of Logistic Regression Results
#  
# Parameters
# logistic regression on test set: Result of LR performed on Test Set
# X_test: X_test obtained from train_test_split
# y_test: y_test obtained from train_test_split
def confusion_matrix_plot(LR, X_test,y_test,flag):
    if (flag==0):
        disp = plot_confusion_matrix(LR, X_test, y_test,cmap=plt.cm.Blues)
        disp.ax_.set_title("Logistic Regression Confusion matrix")
        plt.grid(None) 
        plt.savefig('analysis_visualization/LR_Confusion_Matrix.png')
    elif(flag==1):
        disp = plot_confusion_matrix(LR, X_test, y_test,cmap=plt.cm.Blues)
        disp.ax_.set_title("Logistic Regression Confusion matrix using Re-Sampling")
        plt.grid(None) 
        plt.savefig('analysis_visualization/LR_Confusion_Matrix_resampled.png')
    

# Visualize the accuracy based on number of iterations
#  
# Parameters
# logistic regression results: Dictionary Containing accuracies that need to be plotted
def visualize_LR_accuracy(cv_result,flag):
    if (flag==0):
        fig = plt.figure(1, figsize=(9, 9))
        plt.plot(cv_result)
        plt.title('Accuracy using Logistic Regression')
        plt.ylabel('Model Accuracy %')
        plt.xlabel("No. of iterations")
        sns.set_style("dark")
        plt.savefig('analysis_visualization/Accuracy_LR.png')
    elif(flag==1):
        fig = plt.figure(1, figsize=(9, 9))
        plt.plot(cv_result)
        plt.title('Accuracy using Logistic Regression using Re-Sampling')
        plt.ylabel('Model Accuracy %')
        plt.xlabel("No. of iterations")
        sns.set_style("dark")
        plt.savefig('analysis_visualization/Accuracy_LR_resampled.png')

# Visualize the Logistic Regression accuracy based on removing each column of the dataset at a 
# time and then measuring accuracy
#  
# Parameters
# Logistic Regression: dictionary containing results after individually removing columns
def logistic_regression_visualization(logistic_regression,flag):
    if (flag==0):
        plt.plot(list(logistic_regression.values()), '--', marker='o')
        ax = plt.subplot()
        keys = logistic_regression.keys()
        ax.set_xticklabels(keys, rotation=(25), fontsize=8, ha='right')
        plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), keys)

        plt.title('Logistic Regression')
        plt.xlabel('Columns Removed')
        plt.grid(True)
        plt.ylabel('Accuracy')
        plt.savefig('analysis_visualization/logistic_regression.png')
    elif(flag==1):
        plt.plot(list(logistic_regression.values()), '--', marker='o')
        ax = plt.subplot()
        keys = logistic_regression.keys()
        ax.set_xticklabels(keys, rotation=(25), fontsize=8, ha='right')
        plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), keys)

        plt.title('Logistic Regression Re-Sampled')
        plt.xlabel('Columns Removed')
        plt.grid(True)
        plt.ylabel('Accuracy')
        plt.savefig('analysis_visualization/logistic_regression_resampled.png')

# Apply Logistic Regression Algorithm on the dataset and compare how different 
# approaches in implementing the algorithm impacts the accuracy
#
# The first approach is to find out the best parameters using grid search
#
# In the second approach we try to remove individually one column at a time and try to find out 
# the accuracy respectively. This way we can find out which column is affecting the outcome much or
# indicating the importance of each column
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# df_clean: Original dataset upon which train_test_split was applied
#
# Return: dictionary containing results after individually removing columns
def logistic_regression(X_train, X_test, y_train, y_test, df_clean,flag):

    print("Begin Logistic Regression Analysis")
    # Define parameters for optimization of Logistic Regression
    LR_para = {'C':[0.001, 0.1, 1, 10, 100],'max_iter':[10000]}
    LR_opt=[]
    LR_opt.append((LogisticRegression(), LR_para))
    resultLR=[]
    
    #Checking the accuracy of the Logistic Regression model using Grid Search and Cross Validation 
    for model, para in LR_opt: 

         # 7 Fold Cross-Validation 
        kfold = KFold(7, random_state=0, shuffle=True)

         # Performing Grid Search
        model_grid = GridSearchCV(model, para)

         # Fit data to the model --
        model_grid.fit(X_train,y_train)
         
         # cross validation performed on model obtained from Grid Search
        cv_result = cross_val_score(model_grid, X_train, y_train, cv = kfold, scoring="accuracy")

        # Printing accuracies, best parameters, and best estimator
        print ("Cross Validation Accuracy For LR :- Accuracy: %f SD: %f" % (cv_result.mean(), cv_result.std()))
        print ("Best parameters for Logistic regression :", model_grid.best_params_, model_grid.best_params_['C']) 
        print("Test set score: {:.2f}".format(model_grid.score(X_test, y_test)))
        
    #Evaluating the Confusion Matrix and Classification Report
    LR = LogisticRegression(C=model_grid.best_params_['C'], max_iter=model_grid.best_params_['max_iter'])
    LR.fit(X_train, y_train)
    y_pred_LR = LR.predict(X_test)
    print('Classification Report: \n' + str(cls_report(y_test, y_pred_LR)))
    lr_confusion_matrix = confusion_matrix(y_test, y_pred_LR)
    
    
    if (flag==0):
        #visualize Confusion Matrix
        confusion_matrix_plot(LR, X_test,y_test,0)
        # Visualize Logistic regression accuracy
        visualize_LR_accuracy(cv_result,0)
    elif (flag==1):
        #visualize Confusion Matrix
        confusion_matrix_plot(LR, X_test,y_test,1)
        # Visualize Logistic regression accuracy
        visualize_LR_accuracy(cv_result,1)

    # Store accuracy for no columns removed
    dict = {}
    dict['None'] = model_grid.best_score_

    # Find out which column is most impactful in predicting stop_outcome
    # y = df_clean["stop_outcome"].copy()
    # df_clean = df_clean.drop(columns=['stop_outcome'])

    # # For every column in the dataset, remove the column and train the model, store accuracy
    # for c in df_clean.columns:
    #     print("Removing column", c)
    #     x_t = df_clean.drop(columns=[c])
    #     x_t = pd.get_dummies(x_t)
    #     X_train, X_test, y_train, y_test = train_test_split(x_t, y, test_size=0.2, random_state=0)
    #     for model, para in LR_opt:    
    #         kfold = KFold(7, random_state=0, shuffle=True)
    #         grid_search = GridSearchCV(model, para)
    #         grid_search.fit(X_train,y_train)
    #         cv_result = cross_val_score(grid_search, X_train, y_train, cv = kfold, scoring="accuracy")
    #         dict[c] = grid_search.best_score_

    # # print("Results:\n", dict)
    # print("CV result",cv_result)
    # print("Logistic Regression Completed")

    return dict

# Apply Random Forest classifier algorithm on the dataset and compare how different 
# approaches in implementing the algorithm impacts the accuracy
#
# The first approach is to find out the best parameters using grid search
#
# In the second approach we try to remove individually one column at a time and try to find out 
# the accuracy respectively. This way we can find out which column is affecting the outcome much or
# indicating the importance of each column
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# df_clean: Original dataset upon which train_test_split was applied
#
# Return: dictionary containing results after individually removing columns
def random_forest(X_train, X_test, y_train, y_test, df_clean):
    print("Start Random Forest")
    # Find best parameters to run the model most efficently
    # Set the parameters you want to evaluate
    # param_grid = {'n_estimators': [50,75,100,150,200,250,300,350,400,450,500],
    # 			'max_depth': [None,1,2,3,4,5,10,20,30,50,75,100,150]
    # 			}
    param_grid = {'n_estimators': [250], 'max_depth': [None]}
    dict = {}

    # Create the GridSearch object for the Random Forest classifier passing the parameters
    grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=0),
                               param_grid, cv=5)

    print("Removing no columns")

    # Fit data to the model -- cross validation will be performed during grid search
    grid_search.fit(X_train, y_train)

    # Printing accuracies, best parameters, and best estimator
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print("Best estimator:\n{}".format(grid_search.best_estimator_))
    print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

    # See how well the model is accurately predicting the stop_outcome for each type
    print("Prediction Results:\n", common_utils.find_accuracy_of_each_class(y_test,grid_search.predict(X_test)))

    # Store accuracy for no columns removed
    dict['None'] = grid_search.best_score_

    # Find out which column is most impactful in predicting stop_outcome
    y = df_clean["stop_outcome"].copy()
    df_clean = df_clean.drop(columns=['stop_outcome'])

    # For every column in the dataset, remove the column and train the model, store accuracy
    for c in df_clean.columns:
        print("Removing column", c)
        x_t = df_clean.drop(columns=[c])
        x_t = pd.get_dummies(x_t)
        X_train, X_test, y_train, y_test = train_test_split(x_t, y, test_size=0.2, random_state=0)
        grid_search = GridSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=0),
                                   param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        dict[c] = grid_search.best_score_

        # See how well the model is accurately predicting the stop_outcome for each type
        print("Prediction Results:\n", common_utils.find_accuracy_of_each_class(y_test,grid_search.predict(X_test)))


    print("Results:\n", dict)
    print("Random Forest Completed")

    # Return the results
    return dict

# Visualize the random forest accuracy based on removing each column of the dataset at a 
# time and then measuring accuracy
#  
# Parameters
# random_forest: dictionary containing results after individually removing columns
def random_forest_visualizaton(random_forest):
    # Visualization based on removing individual columns and their respective accuracy
    plt.plot(list(random_forest.values()), '--', marker='o')
    ax = plt.subplot()

    keys = random_forest.keys()
    ax.set_xticklabels(keys, rotation=(25), fontsize=8, ha='right')
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), keys)

    plt.title('Random Forest')
    plt.xlabel('Columns Removed')
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.savefig('analysis_visualization/random_forest.png')

# Apply Decision Tree classifier algorithm on the dataset and compare how different 
# approaches in implementing the algorithm impacts the accuracy
#
# The first approach is to find out the best parameters using the grid search
#
# In the second approach we try to remove individually one column at a time and try to find out 
# the accuracy respectively. This way we can find out which column is affecting the outcome much or
# indicating the importance of each column
#
# The third approach is to find accuracy by varying the depth. The Depth parameter for the Decision classifier
# is varied. For various depths the accuracy is calculated. The increase in the maximum depth causes the
# algorithm to overifit, hence higher depths are not preferred
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# df_clean: Original dataset upon which train_test_split was applied
def decision_tree(X_train, X_test, y_train, y_test, df_clean):
    # Set the parameters you want to evaluate

    params = {'max_leaf_nodes': list(range(2, 30)), 'min_samples_split': [2, 3]}

    # Create the GridSearch object for the Random Forest classifier passing the parameters
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, verbose=1, cv=3)

    # Fit data to the model -- cross validation will be performed during grid search
    grid_search_cv.fit(X_train, y_train)

    # Printing accuracies, best parameters, and best estimator
    print("Best parameters: {}".format(grid_search_cv.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search_cv.best_score_))
    print("Best estimator:\n{}".format(grid_search_cv.best_estimator_))
    print("Test set score: {:.2f}".format(grid_search_cv.score(X_test, y_test)))

    # See how well the model is accurately predicting the stop_outcome for each type
    print("Prediction Results for decision Tree with no column removed:\n", common_utils.find_accuracy_of_each_class(y_test,grid_search_cv.predict(X_test)))

    dict = {}

    # Store accuracy for no columns removed
    dict['None'] = grid_search_cv.best_score_

    # Find out which column is most impactful in predicting stop_outcome
    y = df_clean["stop_outcome"].copy()
    df_clean = df_clean.drop(columns=['stop_outcome'])

    # For every column in the dataset, remove the column and train the model, store accuracy
    list_column_removed = []
    list_accuracy_when_column_removed = []


    for c in df_clean.columns:
        print("Removing column", c)
        x_t = df_clean.drop(columns=[c])
        x_t = pd.get_dummies(x_t)
        X_train, X_test, y_train, y_test = train_test_split(x_t, y, test_size=0.2, random_state=0)
        grid_search = GridSearchCV(DecisionTreeClassifier(max_depth=16), params, cv=5)
        grid_search.fit(X_train, y_train)
        list_column_removed.append(c)
        print("Column Removed:"+str(c)+", Accuracy:"+str(grid_search.best_score_))
        list_accuracy_when_column_removed.append(grid_search.best_score_)

        # print("Prediction Results for coumn {} removed:\n".format(str(c)), common_utils.find_accuracy_of_each_class(y_test,grid_search.predict(X_test)))

    
    # The list of depth which are considered and accuracies are calculated
    list_depths = list(range(1,17))

    list_accuracy = []

    # The prediction and calculation of the accuracy using the decision tree
    for depth in list_depths:
        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth=depth)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        print("Accuracy with depth {} :".format(depth), metrics.accuracy_score(y_test, y_pred))
        acc = metrics.accuracy_score(y_test, y_pred)

        # List of accuracies to plot the data
        list_accuracy.append(acc)

    decison_tree_results = {
    "list_column_removed": list_column_removed,
    "list_accuracy_when_column_removed":list_accuracy_when_column_removed,
    "list_depths":list_depths, 
    "list_accuracy":list_accuracy
    }

    return decison_tree_results

# Visualize the accuracy based on either changing the depth
# or by removing each column of the dataset at a time and then 
# measuring accuracy
#  
# Parameters
# decision_tree_results: Dictionary Containing accuracies that need to be plotted
def decision_tree_visualizaton(decision_tree_results, type):
    plt.close()   
    
    # Visualization based on removing individual columns and their respective accuracy 
    plot_line_chart(decision_tree_results["list_column_removed"], decision_tree_results["list_accuracy_when_column_removed"], "Column", "Accuracy", "Column Removed vs Accuracy", "analysis_visualization/decision_tree_column_{}.png".format(type))   
    plt.close()

    # Visualization based on varying depths and their respective accuracy 
    plot_line_chart(decision_tree_results["list_depths"], decision_tree_results["list_accuracy"], "Maximum Depth", "Accuracy", "Depth vs Accuracy", "analysis_visualization/decision_tree_depth_{}.png".format(type))



# Apply KNN algorithm on the dataset and compare how different approaches in implementing the algorithm impacts the accuracy
# The first approach is to apply KNN on the entire dataset by selecting n number of neighbours
# The second approach is to reduce the dataset into n number of features and then apply KNN on the dataset
# The third approach is to find accuracy by using cross_val_score with 5 folds on KNN
# Finally, drop one column at a time and find the importance of each column in the dataset
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# df_clean: Original dataset upon which train_test_split was applied
def k_neighbors_classifier(X_train, X_test, y_train, y_test, df_clean):

    fig, ax = plt.subplots()

    # KNN accuracy for n neighbors on entire dataset
    X_axis, Y_axis = knn_analysis.knn_accuracy_on_entire_dataset(X_train, X_test, y_train, y_test)
    # Test Results
    # print (Y_axis)
    # Y_axis = [0.8935, 0.8824, 0.8693, 0.8602, 0.8574, 0.8473, 0.842, 0.8312, 0.8302, 0.8213, 0.82, 0.8111, 0.8128, 0.8077, 0.8055, 0.7971, 0.7922, 0.7885, 0.785, 0.7813, 0.7769, 0.7723, 0.7715, 0.7673]
    # Plot the data as a line chart and mark it in blue
    plt.plot(X_axis, Y_axis, label='KNN accuracy for n neighbors on entire dataset (82 features)', c="blue")

    # Apply cross_val_score
    X_axis, Y_axis = knn_analysis.knn_apply_cross_val_score(df_clean)
    # Test Results
    # print (Y_axis)
    # Y_axis = [0.8933, 0.88368, 0.8752600000000001, 0.8701599999999999, 0.8649799999999999, 0.8580399999999999, 0.8549200000000001, 0.84672, 0.8450799999999999, 0.8377399999999999, 0.8348000000000001, 0.8295999999999999, 0.8274199999999998, 0.8228199999999999, 0.81822, 0.81458, 0.8116, 0.80554, 0.80242, 0.79642, 0.79224, 0.7880400000000001, 0.78308, 0.7794399999999999]
    # Plot the data as a line chart and mark it in green
    plt.plot(X_axis, Y_axis, label='cross_val_score with 5 fold for n neighbors', c="green")

    plt.xlabel("n neighbours")
    plt.ylabel("accuracy score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("analysis_visualization/KNN.png") 

    fig, ax = plt.subplots()
    # Apply PCA and compare
    X_axis, Y_axis = knn_analysis.apply_pca_and_compare(X_train, X_test, y_train, y_test)
    # Test Results
    # print (Y_axis)
    # Y_axis = [0.2859, 0.4373, 0.6663, 0.672, 0.6861, 0.6962, 0.7128, 0.7109, 0.7116, 0.7127, 0.7141, 0.725, 0.7317, 0.7335, 0.7385, 0.7345, 0.7366, 0.7398, 0.7309, 0.7403, 0.7368, 0.7339, 0.7395, 0.7356]
    # Plot the data as a line chart and mark it in red
    plt.plot(X_axis, Y_axis, label='KNN accuracy after applying PCA and reducing to n components', c="red")

    plt.xlabel("n components")
    plt.ylabel("accuracy score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("analysis_visualization/PCA.png") 

    # drop one column at a time and find the importance of each column in the dataset
    X_axis, Y_axis = knn_analysis.knn_remove_columns_and_find_accuracy(df_clean)
    # Test Results
    # print (X_axis)
    # print (Y_axis)
    # X_axis = ['None', 'stop_year', 'stop_month', 'stop_date', 'stop_hour', 'driver_gender', 'drivers_age_bucket', 'drivers_race', 'stop_duration', 'is_arrested', 'drugs_related_stop', 'violations_raw', 'search_score']
    # Y_axis = [0.8602, 0.8433, 0.8659, 0.8608, 0.8619, 0.8637, 0.8676, 0.8629, 0.8546, 0.8179, 0.8606, 0.8375, 0.8602]
    plot_line_chart(X_axis, Y_axis, "column", "Accuracy", "KNN accuracy for on removing each feature", "analysis_visualization/KNN_columns.png")


# Since our dataset is predominatly filled with citations, we need to resample our data so that classification models can learn better
# This method will upscale and downscale columns accordingly so that there is an even distribution of outcomes for all models to predict 
# 
# Parameters
# dataframe: dataframe containing processed data
# n_samples: number of samples of each outcome that needs to be generated. Value has to be between 5293 and 77005
#
# Return: dataframe which has been resampled
def resample_data(dataframe, n_samples):
    if n_samples < 5293 or n_samples > 77005:
        return None

    df_citation = dataframe[dataframe['stop_outcome'] == 'Citation']
    df_warning = dataframe[dataframe['stop_outcome'] == 'Warning']
    df_arrest_driver = dataframe[dataframe['stop_outcome'] == 'Arrest Driver']
    df_no_action = dataframe[dataframe['stop_outcome'] == 'No Action']
    df_arrest_passenger = dataframe[dataframe['stop_outcome'] == 'Arrest Passenger']
     
    # Resample all values
    df_citation = resample(df_citation, replace=False, n_samples=n_samples, random_state=123)
    df_warning = resample(df_warning, replace=True, n_samples=n_samples, random_state=123)
    df_arrest_driver = resample(df_arrest_driver, replace=True, n_samples=n_samples, random_state=123)
    df_no_action = resample(df_no_action, replace=True, n_samples=n_samples, random_state=123)
    df_arrest_passenger = resample(df_arrest_passenger, replace=True, n_samples=n_samples, random_state=123)
     
    # Combine minority class with downsampled majority class
    df_resampled = pd.concat([df_citation, df_warning, df_arrest_driver, df_no_action, df_arrest_passenger])
    
    return df_resampled

if __name__ == "__main__":
    filename = "datasets/police_project.csv"

    # Converting the CSV to a panda dataframe
    dataframe = read_file(filename)

    # Data preprocessing and visualization

    # Hari Analysis
    print("transforming data columns_1_4")
    dataframe = transform_data_columns_1_4(dataframe)
    # Akash's Anaylsis
    print("transforming data columns_5_8")
    dataframe = transform_data_columns_5_8(dataframe)
    # Jayasurya Analysis
    print("transforming data columns_9_12")
    dataframe = transform_data_columns_9_12(dataframe)
    # Anthony Analysis
    print("transforming data columns_13_15")
    dataframe = transform_data_columns_13_15(dataframe)

    print("Visualizing data. Commented this part to save time")
    # # Hari visualization
    # visualize_data_columns_1_4(dataframe)
    # # Akash's visualization
    # visualize_data_columns_5_8(dataframe)
    # # Jayasurya visualization
    # visualize_data_columns_9_12(dataframe)
    # # Anthony visualization
    # visualize_data_columns_13_15(dataframe)

    # Last minute catches to finalize preprocessing
    print("General Preprocessing")
    dataframe_orig = general_preprocessing(dataframe)

    filename = 'datasets/processed_data_orig.csv'
    print("Saving dataframe into", filename)
    save_to_csv(dataframe_orig, filename)

    # Perform a test train split to train our model
    print("Performing a test train split to train our model")
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = common_utils.test_train_split(dataframe_orig)

    # Since our dataset is predominantly filled with citations, we are downsampling our data so that there is better learning
    print("Resampling data")
    dataframe = resample_data(dataframe_orig, 10000)

    filename = 'datasets/resampled_data.csv'
    print("Saving resampled data into", filename)
    save_to_csv(dataframe, filename)

    # Perform a test train split to train our model
    print("Performing a test train split to train our model")
    X_train, X_test, y_train, y_test = common_utils.test_train_split(dataframe)

    # Decision Trees
    # Commented out since still a work in progress
    print("decision_tree without resampling data")
    decision_tree_results = decision_tree(X_train_orig, X_test_orig, y_train_orig, y_test_orig, dataframe_orig)

    # Displaying results from running decision tree
    decision_tree_visualizaton(decision_tree_results, "without_resampling")

    # Decision Trees
    # Commented out since still a work in progress
    print("decision_tree with resampling data")
    # decision_tree_results = decision_tree(X_train, X_test, y_train, y_test, dataframe)

    
    # Displaying results from running decision tree
    # decision_tree_visualizaton(decision_tree_results, "with_resampling")


    # Random Forest
    # random_forest_results = random_forest(X_train, X_test, y_train, y_test, dataframe)

    # Results from running random forest (so you don't have to run the method)
    # Comment out the following line if you decide to run the random_forest method and get the accuracies from there
    # random_forest_results = {'None': 0.9567625, 'stop_year': 0.9503375000000001, 'stop_month': 0.9529125, 
    # 'stop_date': 0.949075, 'stop_hour': 0.9527749999999999, 'driver_gender': 0.9554874999999999, 
    # 'drivers_age_bucket': 0.9533250000000001, 'drivers_race': 0.9558625000000001, 'stop_duration': 0.9561125, 
    # 'is_arrested': 0.9446625, 'drugs_related_stop': 0.9561249999999999, 'violations_raw': 0.9544750000000001, 
    # 'search_score': 0.9555875}
    # Displaying results from running random forest
    # random_forest_visualizaton(random_forest_results)



    #Logistic Regression
    # Commented out since still a work in progress
    # print("Logistic Regression")
    logistic_regression_results = logistic_regression(X_train_orig, X_test_orig, y_train_orig, y_test_orig, dataframe_orig , 1)
     #RESULTS OF LOGISTIC REGRESSION:
    # {'None': 0.9280428913411332, 'stop_year': 0.9279991839879596, 'stop_month': 0.9281448557161394,
    # 'stop_date': 0.928086581720577, 'stop_hour': 0.9280574569226638, 'driver_gender': 0.9279554893650832, 
    # 'drivers_age_bucket': 0.9279700581291884, 'drivers_race': 0.9279846226498613, 'stop_duration': 0.9280428849759845,
    # 'is_arrested': 0.8978908252908356, 'drugs_related_stop': 0.9279991924748245, 'violations_raw': 0.9275039340596075,
    # 'search_score': 0.9280574579835221}
    logistic_regression_visualization(logistic_regression_results,1)

    #RESAMPLING
    logistic_regression_results_resampled = logistic_regression(X_train, X_test, y_train, y_test, dataframe , 1)
    # logistic_regression_results_resampled {'None': 0.7005750000000001, 'stop_year': 0.677275, 'stop_month': 0.6999749999999999, 
    # 'stop_date': 0.6905749999999999, 'stop_hour': 0.6992, 'driver_gender': 0.689575, 'drivers_age_bucket': 0.698325,
    # 'drivers_race': 0.696775, 'stop_duration': 0.68965, 'is_arrested': 0.566025, 'drugs_related_stop': 0.69585, 
    # 'violations_raw': 0.588125, 'search_score': 0.69915}
    logistic_regression_visualization(logistic_regression_results_resampled,1)

   

   
    
    
    # # K Nearest Neighbours
    # print ("k_neighbors_classifier")
    # k_neighbors_classifier(X_train, X_test, y_train, y_test, dataframe)
    # print ("k_neighbors_classifier performance without resampling data")
    # knn_analysis.knn_find_accuracy_of_each_class(X_train_orig, X_test_orig, y_train_orig, y_test_orig)
    # print ("k_neighbors_classifier performance with resampling data")
    # knn_analysis.knn_find_accuracy_of_each_class(X_train, X_test, y_train, y_test)
    
