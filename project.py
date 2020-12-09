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
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import classification_report as cls_report
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
# x: values that needs to be present in x axis
# y: corresponding y values that needs to be plotted
# x_label: label that needs to be marked on x axis
# y_label: label that needs to be marked on y axis
# title: title for the graph
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
    plot_pie_chart(df, 'drivers_race', 'priliminary_visualization/drivers_race.png')
    plot_bar_graph(df, 'violations_raw', 'priliminary_visualization/violations_raw.png')
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


def test_train_split(dataframe):
    # Set X for test train split and use get_dummies for one hot encoding
    X = dataframe.drop(columns=["stop_outcome"])
    X = pd.get_dummies(X)

    # Set y for test train split
    y = dataframe["stop_outcome"].copy()

    # Perform the test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def logistic_regression(X_train, X_test, y_train, y_test, df_clean):
    print("Begin Logistic Regression Analysis")
    # Define parameters for optimization of Logistic Regression
    LR_para = {'C':[0.001, 0.1, 1, 10, 100],'max_iter':[1000000]}
    LR_opt=[]
    LR_opt.append((LogisticRegression(), LR_para))
    resultLR=[]
    print("yes")
    #Checking the accuracy of the Logistic Regression model using Grid Search and Cross Validation 
    for model, para in LR_opt:    
        kfold = KFold(2, random_state=0, shuffle=True)
        model_grid = GridSearchCV(model, para)
        model_grid.fit(X_train,y_train)
        cv_result = cross_val_score(model_grid, X_train, y_train, cv = kfold, scoring="accuracy")
        print ("Cross Validation Accuracy For LR :- Accuracy: %f SD: %f" % (cv_result.mean(), cv_result.std()))
        print ("Best parameters for Logistic regression :", model_grid.best_params_) 
        print("Test set score: {:.2f}".format(model_grid.score(X_test, y_test)))
        print('Classification Report: \n' + str(cls_report(y_test, cv_result)))
        confusion_matrix = confusion_matrix(y_test, cv_result)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion_matrix,cmap=plt.cm.Blues)
    ax.grid(False)
    ax.set_xlabel('Predicted outputs')
    ax.set_ylabel('Actual outputs')

    plt.title("Logistic Regression Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black',fontsize=50)
    plt.savefig('Confusion_Matrix_LR.png')

    return cv_result


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

    print("Results:\n", dict)
    print("Random Forest Completed")

    # Return the results
    return dict


def random_forest_visualizaton(random_forest):
    plt.plot(list(random_forest.values()), '--', marker='o')
    ax = plt.subplot()

    keys = random_forest.keys()
    ax.set_xticklabels(keys, rotation=(25), fontsize=8, ha='right')
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), keys)

    plt.title('Random Forest')
    plt.xlabel('Columns Removed')
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.savefig('random_forest.png')


def decision_tree(X_train, X_test, y_train, y_test, df_clean):
    # Set the parameters you want to evaluate

    params = {'max_leaf_nodes': list(range(2, 8)), 'min_samples_split': [2, 3]}

    # Create the GridSearch object for the Random Forest classifier passing the parameters
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, verbose=1, cv=3)

    # Fit data to the model -- cross validation will be performed during grid search
    grid_search_cv.fit(X_train, y_train)

    # Printing accuracies, best parameters, and best estimator
    print("Best parameters: {}".format(grid_search_cv.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search_cv.best_score_))
    print("Best estimator:\n{}".format(grid_search_cv.best_estimator_))
    print("Test set score: {:.2f}".format(grid_search_cv.score(X_test, y_test)))

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
        grid_search = GridSearchCV(DecisionTreeClassifier(max_depth=5), params, cv=5)
        grid_search.fit(X_train, y_train)
        list_column_removed.append(c)
        print("Column Removed:"+str(c)+", Accuracy:"+str(grid_search.best_score_))
        list_accuracy_when_column_removed.append(grid_search.best_score_)
    
    # The list of depth which are considered and accuracies are calculated
    list_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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


def decision_tree_visualizaton(decision_tree_results):

    # Visualization based on removing individual columns and their respective accuracy 
    plt.plot(decision_tree_results["list_column_removed"], decision_tree_results["list_accuracy_when_column_removed"], label='Column Removed vs Accuracy')

    plt.xlabel('Column Removed')  # Label x-axis
    plt.ylabel('Accuracy')  # Label y-axis
    plt.grid(True)
    plt.legend()  # Show plot labels as legend
    plt.ylim(ymin=0.8)
    plt.savefig('decision_tree_column.png')  # Save graph

    plt.close()

    # Visualization based on varying depths and their respective accuracy 
    plt.plot(decision_tree_results["list_depths"], decision_tree_results["list_accuracy"], label='Depth vs Accuracy')

    plt.xlabel('Maximum Depth')  # Label x-axis
    plt.ylabel('Accuracy')  # Label y-axis
    plt.grid(True)
    plt.legend()  # Show plot labels as legend
    plt.ylim(ymin=0.9)
    plt.savefig('decision_tree_depth.png')  # Save graph


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
    # Y_axis = [0.9090538336052202, 0.8994406898158938, 0.9086460032626428, 0.9042763924493125, 0.9079468655325099, 0.905558144954556, 0.9067233745047775, 0.905499883477045, 0.9057911908646004, 0.9050920531344675, 0.905499883477045, 0.9048590072244233, 0.90468422279189, 0.9040433465392682, 0.9040433465392682, 0.9035772547191797, 0.9034607317641575, 0.9027615940340247, 0.9025285481239804, 0.9023537636914473, 0.9021207177814029, 0.9015963644838033, 0.9017711489163365, 0.9012467956187369, 0.9010720111862037, 0.900547657888604, 0.9008389652761594, 0.900547657888604, 0.9003146119785598, 0.9000233045910044, 0.9002563505010487, 0.9000233045910044, 0.9000815660685155, 0.8998485201584712, 0.8994989512934048, 0.8994406898158938, 0.8993241668608716, 0.8987415520857609, 0.8988580750407831, 0.8987415520857609, 0.8986832906082498, 0.8983919832206945, 0.8981589373106502, 0.8982754602656723, 0.8982754602656723, 0.897984152878117, 0.898042414355628, 0.8978093684455838, 0.8976928454905616]
    # Plot the data as a line chart and mark it in blue
    plt.plot(X_axis, Y_axis, label='KNN accuracy for n neighbors on entire dataset (82 features)', c="blue")

    # Apply PCA and compare
    X_axis, Y_axis = knn_analysis.apply_pca_and_compare(X_train, X_test, y_train, y_test)
    # Y_axis = [0.6000932183640177, 0.7787229084129573, 0.8856909811232813, 0.8885457935213237, 0.8825448613376835, 0.8833605220228385, 0.8810300629223957, 0.8844674900955488, 0.8848170589606152, 0.8864483803309252, 0.8882544861337683, 0.8889536238639012, 0.8880797017012352, 0.8874388254486134, 0.8867979491959916, 0.8892449312514565, 0.888720577953857, 0.8873223024935912, 0.8881379631787462, 0.8867979491959916, 0.887205779538569, 0.8877301328361688, 0.8892449312514565, 0.8889536238639012, 0.8876136098811466, 0.8870309951060359, 0.8884875320438126, 0.8888953623863901, 0.8998485201584712, 0.8983337217431834, 0.8999067816359823, 0.8995572127709158, 0.8991493824283384, 0.8997902586809602, 0.8992659053833605, 0.8997319972034491, 0.9003146119785598, 0.8994989512934048, 0.8991493824283384, 0.8992076439058495, 0.8989163365182941, 0.8992076439058495, 0.8981589373106502, 0.8988580750407831, 0.8998485201584712, 0.9000815660685155, 0.8989745979958051, 0.8991493824283384, 0.8993241668608716]
    # Plot the data as a line chart and mark it in red
    plt.plot(X_axis, Y_axis, label='KNN accuracy after applying PCA and reducing to n components', c="red")

    # Apply cross_val_score
    X_axis, Y_axis = knn_analysis.knn_apply_cross_val_score(df_clean)
    # Y_axis = [0.86772711648944, 0.8294348266933245, 0.8745091963596476, 0.8566798345178234, 0.882048839875636, 0.8715376813602946, 0.8871296903055772, 0.8800795287465333, 0.8899731260558615, 0.8854517036732, 0.8931661544339489, 0.890194600463708, 0.8949025109019156, 0.8923970915246893, 0.8963009441467522, 0.8942965915358007, 0.8972215828989965, 0.8956950076714669, 0.8980722755609056, 0.8971866450226792, 0.8990861288751297, 0.8984452127011104, 0.8991444245709814, 0.8987831786971668, 0.8995290035510842, 0.8992143782653922, 0.8999718478107364, 0.8995989505919287, 0.9002631922681607, 0.899727156258756, 0.900426353870985, 0.8999835400276358, 0.9000884382519302, 0.9001700081224835, 0.9003215031721636, 0.9000534899200089, 0.900473008677448, 0.9002399561138377, 0.9005079750690493, 0.9004031006074916, 0.9005312910661674, 0.9001700508954092, 0.9004264261097042, 0.9002050153859917, 0.9004264384663271, 0.9002283389871854, 0.9002399912826876, 0.9001584299667194, 0.9001118217352201]
    # Plot the data as a line chart and mark it in green
    plt.plot(X_axis, Y_axis, label='cross_val_score with 5 fold for n neighbors', c="green")

    plt.xlabel("n")
    plt.ylabel("score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("analysis_visualization/KNN.png") 

    # drop one column at a time and find the importance of each column in the dataset
    X_axis, Y_axis = knn_analysis.knn_remove_columns_and_find_accuracy(df_clean)
    # X_axis = ['None', 'stop_year', 'stop_month', 'stop_date', 'stop_hour', 'driver_gender', 'drivers_age_bucket', 'drivers_race', 'stop_duration', 'is_arrested', 'drugs_related_stop', 'violations_raw', 'search_score']
    # Y_axis = [0.9042763924493125, 0.9074807737124213, 0.9080633884875321, 0.9046259613143789, 0.9116173386157073, 0.9060242367746446, 0.9077138196224657, 0.9069564204148217, 0.903810300629224, 0.8908762526217665, 0.9044511768818457, 0.9063155441622, 0.9076555581449546]
    plot_line_chart(X_axis, Y_axis, "column", "Accuracy", "KNN accuracy for on removing each feature", "analysis_visualization/KNN_columns.png")


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
    dataframe = general_preprocessing(dataframe)

    filename = 'datasets/processed_data.csv'
    print("Saving dataframe into", filename)
    save_to_csv(dataframe, filename)

    # Perform a test train split to train our model
    print("Performing a test train split to train our model")
    X_train, X_test, y_train, y_test = test_train_split(dataframe)

    # Random Forest
    # Commented out since still a work in progress
    # random_forest_results = random_forest(X_train, X_test, y_train, y_test, dataframe)

    # Results from running random forest (so you don't have to run the method)
    # Comment out this line if you decide to run the random_forest method and get the accuracies from there
    # random_forest_results = {'stop_year': 0.9225951283381028, 'stop_month': 0.9195216378715305,
    # 'stop_date': 0.9111169567927885, 'stop_hour': 0.919958632899766,
    # 'driver_gender': 0.9260764349312349, 'drivers_age_bucket': 0.9211530328633148,
    # 'drivers_race': 0.9256394908241876, 'stop_duration': 0.9267173438574698,
    # 'is_arrested': 0.8986191255845395, 'drugs_related_stop': 0.92706694009223,
    # 'violations_raw': 0.9240226264056173, 'search_score': 0.9265280072104403,
    # 'None': 0.9267901887388534}
    # Displaying results from running random forest
    # random_forest_visualizaton(random_forest_results)

    # # Decision Trees
    # # Commented out since still a work in progress
    # print("decision_tree")
    # decision_tree_results = decision_tree(X_train, X_test, y_train, y_test, dataframe)

    # # Displaying results from running decision tree
    # decision_tree_visualizaton(decision_tree_results)

    #Logistic Regression
    # Commented out since still a work in progress
    # print("Logistic Regression")
    # logistic_regression_results = logistic_regression(X_train, X_test, y_train, y_test, dataframe)

    # K Nearest Neighbours
    # print ("k_neighbors_classifier")
    # KNN_results = k_neighbors_classifier(X_train, X_test, y_train, y_test, dataframe)
