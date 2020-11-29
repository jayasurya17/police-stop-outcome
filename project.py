import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def read_file(filename):
  	return pd.read_csv(filename)


def group_and_plot_pivot_graph(df, columns, filename):
	grouped_multiple = df.groupby(columns).size().reset_index()
	grouped_multiple.columns = columns + ['count']
	# fig, ax = plt.subplots()
	fig = plt.figure()
	grouped_multiple.pivot(columns[0], columns[1], 'count').plot(kind='bar')

	plt.ylabel('count')
	plt.legend(title=columns[1], bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.savefig(filename)


def plot_pie_chart(df, column, filename):
	fig, ax = plt.subplots()
	df[column].value_counts().plot(ax=ax, kind='pie')
	plt.tight_layout()
	plt.savefig(filename)


def plot_bar_graph(df, column, filename):
	fig, ax = plt.subplots()
	df[column].value_counts().plot(ax=ax, kind='bar')
	plt.xlabel(column)
	plt.ylabel('count')
	plt.tight_layout()
	plt.savefig(filename)


def plot_kde_graph(df, column, filename):
	fig, ax = plt.subplots()
	df[column].plot(ax=ax, kind='kde')
	plt.xlabel(column)
	plt.ylabel('Probability Distribution')
	plt.tight_layout()
	plt.savefig(filename)


def plot_hist_graph(df, column, filename):
	fig, ax = plt.subplots()
	df[column].plot(ax=ax, kind='hist')
	plt.xlabel(column)
	plt.ylabel('count')
	plt.tight_layout()
	plt.savefig(filename)


def transform_data_columns_1_4(dataframe):
	#The county name column can be dropped cause it is completely null
	dataframe=dataframe.drop(columns=['county_name'], axis=1, inplace=False)

	# The gender dataset contains nulls. Here those nulls are discarded cause 
	# the dataset is large and the nulls consitute about 5%. Also replacing the
	# null data with frequent values might mislead the prediction
	dataframe = dataframe.dropna(subset=['driver_gender'], inplace=False)

	#Create a colum that gives only the year of stop, which can be used
	#visualization of other columns with year
	dataframe['stop_year'] = dataframe['stop_date'].apply(lambda x: int(x.split('-')[0]))
  
	return dataframe


def transform_data_columns_5_8(df):

    #To clean the driver's age column and treat null values, we can replace it with the average value
	#of age in column where there is an incorrect value, null value or age less than 16.	
	df["drivers_age_new"]= df["driver_age"]
	    
	#To clean the driver's race column and treat null values, we can replace it with the most frequent value
	#of race in column
	df["drivers_race"]= df["driver_race"].fillna(df["driver_race"].mode())
	df=df.drop(columns=["driver_race"], axis=1, inplace=False)
    
	#To clean the violation column and treat null values, we can replace it with the most frequent value
	#of violation in column
	df["violations_raw"]= df["violation_raw"].fillna(df["violation_raw"].mode())
	df=df.drop(columns=["violation_raw"], axis=1, inplace=False)
	
	x=0
	y=0
	z=int(df["driver_age"].mean())
 
	for i in df.index:
		x= df["stop_year"][i]-df["driver_age_raw"][i]
		if(x<16 or df["driver_age"][i]==np.nan or x!= df["driver_age"][i]):
			y=y+1
			df["drivers_age_new"][i]=z
								
	# There are 2 similar columns that is driver's year of birth and driver's age 
    # which provide converging information, pre-pruning is performed and driver's year of birth is
    # dropped as driver's age is sufficient for analysis.
    # It also helps in removing unneccessary branches before performing classification
	df=df.drop(columns=["driver_age"], axis=1, inplace=False)
	df=df.drop(columns=['driver_age_raw'], axis=1, inplace=False)

	return df
    
    
def visualize_data_columns_1_4(dataframe):  
  #The statistics of the first four coumns are as follows

  #Column 1 -> Stop Date - The year on which the person is stopped for each gender
  plot_bar_graph(dataframe,'stop_year','priliminary_visualization/stop_year.png')
  group_and_plot_pivot_graph(dataframe, ['stop_year', 'driver_gender'], 'priliminary_visualization/gender_with_year.png')


  #Column 2 -> Stop Time - The hour on which the person is stopped for each gender
  plot_bar_graph(dataframe,'stop_hour','priliminary_visualization/stop_hour.png')
  group_and_plot_pivot_graph(dataframe, ['stop_hour', 'driver_gender'], 'priliminary_visualization/gender_with_hour.png')

  #Column 3 -> Gender 
  print(dataframe['driver_gender'].value_counts())
  plot_pie_chart(dataframe,'driver_gender','priliminary_visualization/driver_gender.png')

  return dataframe


def visualize_data_columns_5_8(df):
	plot_pie_chart(df, 'drivers_race', 'priliminary_visualization/drivers_race.png')
	plot_bar_graph(df, 'violations_raw', 'priliminary_visualization/violations_raw.png')
	plot_kde_graph(df, 'drivers_age_new', 'priliminary_visualization/drivers_age.png')
	plot_hist_graph(df, 'drivers_age_new', 'priliminary_visualization/drivers_hist_age.png')


def calculate_search_score(weights, search_type):
	search = search_type.split(',')
	total_weight = 0
	for value in search:
		total_weight += pow(2, weights[value])
	return total_weight


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


def visualize_data_columns_9_12(df):
	plot_pie_chart(df, 'search_conducted', 'priliminary_visualization/search_conducted.png')

	df1 = df[df.search_type != 'None']
	s = df1['search_type'].str.split(',', expand=True).stack()
	index = s.index.get_level_values(0)
	df1 = df1.loc[index].copy()
	df1['search_type'] = s.values
	plot_bar_graph(df1, 'search_type', 'priliminary_visualization/search_type.png')

	group_and_plot_pivot_graph(df, ['stop_hour', 'violation'], 'priliminary_visualization/violation.png')
	
	group_and_plot_pivot_graph(df, ['violation', 'stop_outcome'], 'priliminary_visualization/stop_outcome.png')


def visualize_data_columns_13_15(dataframe):
	plot_pie_chart(dataframe, 'is_arrested', 'priliminary_visualization/is_arrested.png')
	plot_pie_chart(dataframe, 'stop_duration', 'priliminary_visualization/stop_duration.png')
	plot_pie_chart(dataframe, 'drugs_related_stop', 'priliminary_visualization/drugs_related_stop.png')

	
def general_preprocessing(dataframe):
  	# Remove unwanted column
	del dataframe['search_conducted']
	del dataframe['search_type']

	dataframe.dropna()
	print(dataframe.count())

	return dataframe

def test_train_split(dataframe):
	# Set X for test train split and use get_dummies for one hot encoding
	X = dataframe.drop(columns=["stop_outcome"])
	X = pd.get_dummies(X)

	# Set y for test train split
	y = dataframe["stop_outcome"].copy()

	# Perform the test train split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	return X_train, X_test, y_train, y_test
    
def random_forest(X_train, X_test, y_train, y_test, df_clean):
	print("Start Random Forest")
	# Set the parameters you want to evaluate 
	# param_grid = {'n_estimators': [50,75,100,150,200,250,300,350,400,450,500],
	# 			'max_depth': [None,1,2,3,4,5,10,20,30,50,75,100,150]
	# 			}
	param_grid = {'n_estimators': [200],'max_depth': [None]}			

	# Create the GridSearch object for the Random Forest classifier passing the parameters
	grid_search = GridSearchCV(RandomForestClassifier(n_jobs= -1, class_weight="balanced", random_state=0), 
							param_grid, cv=5)

	# Fit data to the model -- cross validation will be performed during grid search
	grid_search.fit(X_train, y_train)

	# Printing accuracies, best parameters, and best estimator
	print("Best parameters: {}".format(grid_search.best_params_))
	print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
	print("Best estimator:\n{}".format(grid_search.best_estimator_))
	print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

	dict = {}

	# For every column in the dataset, remove the column and train the model, store accuracy
	for c in df_clean.columns:
		print("Removing column", c)
		x_t = df_clean.drop(columns=[c])
		x_t = pd.get_dummies(x_t)
		X_train, X_test, y_train, y_test = train_test_split(x_t, dataframe["stop_outcome"].copy(), test_size=0.2, random_state=0)
		grid_search = GridSearchCV(RandomForestClassifier(n_jobs= -1, class_weight="balanced", random_state=0), 
								param_grid, cv=5)
		grid_search.fit(X_train, y_train)
		dict[c] = grid_search.best_score_

	# Same for removing no columns
	x_t = pd.get_dummies(df_clean)
	X_train, X_test, y_train, y_test = train_test_split(x_t, dataframe["stop_outcome"].copy(), test_size=0.2, random_state=0)
	print("Removing no columns")
	grid_search = GridSearchCV(RandomForestClassifier(n_jobs= -1, class_weight="balanced", random_state=0), 
							param_grid, cv=5)
	grid_search.fit(X_train, y_train)
	dict['None'] = grid_search.best_score_

	print(dict)
	print("Random Forest Completed")

	# Return the results
	return dict

if __name__ == "__main__":

	filename = "police_project.csv"

	# Converting the CSV to a panda dataframe
	dataframe = read_file(filename)

	# Data preprocessing and visualization

	# Hari Analysis
	dataframe = transform_data_columns_1_4(dataframe)
	# Akash's Anaylsis
	dataframe=transform_data_columns_5_8(dataframe)
	# Jayasurya Analysis
	dataframe = transform_data_columns_9_12(dataframe)
	# Anthony Analysis
	dataframe = transform_data_columns_13_15(dataframe)
	
	# Hari visualization
	visualize_data_columns_1_4(dataframe)
	# Akash's visualization
	visualize_data_columns_5_8(dataframe)
	# Jayasurya visualization
	visualize_data_columns_9_12(dataframe)
	# Anthony visualization
	visualize_data_columns_13_15(dataframe)

	# Last minute catches to finalize preprocessing
	dataframe = general_preprocessing(dataframe)

	# Perform a test train split to train our model
	X_train, X_test, y_train, y_test = test_train_split(dataframe)

	# Random Forest
	# Commented out since still a work in progress
	# random_forest_results = random_forest(X_train, X_test, y_train, y_test, dataframe)

	# Results from running random forest (since it will take a few hours to run)
	# random_forest_results = {'stop_date': 1.0, 'stop_time': 1.0, 'driver_gender': 1.0, 'violation': 1.0, 
	# 						'stop_outcome': 0.9281011579106888, 'is_arrested': 1.0, 'stop_duration': 1.0, 
	# 						'drugs_related_stop': 0.999985433357611, 'stop_year': 1.0, 'drivers_age_new': 0.999985433357611, 
	# 						'drivers_race': 1.0, 'violations_raw': 1.0, 'stop_hour': 0.999985433357611, 'search_score': 1.0, 
	# 						'None': 1.0}