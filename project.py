import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
	df[column].value_counts().plot(ax=ax, kind='kde')
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
	# There are 2 similar columns that is driver's year of birth and driver's age 
    # which provide converging information, pre-pruning is performed and driver's year of birth is
    # dropped as driver's age is sufficient for analysis.
    # It also helps in removing unneccessary branches before performing classification
	df=df.drop(columns=['driver_age_raw'], axis=1, inplace=False)

    #To clean the driver's age column and treat null values, we can replace it with the average value
	#of age in column
	df["drivers_age"]= df["driver_age"].fillna(df["driver_age"].mean())
	df=df.drop(columns=["driver_age"], axis=1, inplace=False)
    
	#To clean the driver's race column and treat null values, we can replace it with the most frequent value
	#of race in column
	df["drivers_race"]= df["driver_race"].fillna(df["driver_race"].mode())
	df=df.drop(columns=["driver_race"], axis=1, inplace=False)
    
	#To clean the violation column and treat null values, we can replace it with the most frequent value
	#of violation in column
	df["violations_raw"]= df["violation_raw"].fillna(df["violation_raw"].mode())
	df=df.drop(columns=["violation_raw"], axis=1, inplace=False)
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
	plot_kde_graph(df, 'drivers_age', 'priliminary_visualization/drivers_age.png')

def transform_data_columns_9_12(df): 
	# Columns
	# violation	
	# search_conducted	
	# search_type	
	# stop_outcome

	# Get the hour of day at which stop happened 
	df['stop_hour'] = df['stop_time'].apply(lambda x: int(x.split(':')[0]))

	# Remove unwanted column
	plot_pie_chart(df, 'search_conducted', 'priliminary_visualization/search_conducted.png')
	del df['search_conducted']
	
	# Fill the empty values with string 'None' and make all the values atomic by splitting the comma separated search types
	df['search_type'].fillna('None', inplace=True)
	s = df['search_type'].str.split(',', expand=True).stack()
	index = s.index.get_level_values(0)
	df = df.loc[index].copy()
	df['search_type'] = s.values

	# Drop rows where the outcome is not avaliable
	df = df[df.stop_outcome != 'N/D']

	return df


def visualize_data_columns_9_12(df):
	df1 = df[df.search_type != 'None']
	plot_bar_graph(df1, 'search_type', 'priliminary_visualization/search_type.png')
	group_and_plot_pivot_graph(df, ['stop_hour', 'violation'], 'priliminary_visualization/violation.png')
	group_and_plot_pivot_graph(df, ['violation', 'stop_outcome'], 'priliminary_visualization/stop_outcome.png')

def visualize_data_columns_13_15(dataframe):
	plot_pie_chart(dataframe, 'is_arrested', 'priliminary_visualization/is_arrested.png')
	plot_pie_chart(dataframe, 'stop_duration', 'priliminary_visualization/stop_duration.png')
	plot_pie_chart(dataframe, 'drugs_related_stop', 'priliminary_visualization/drugs_related_stop.png')

	
def general_preprocessing(dataframe):
  
	# Define columns that are already mostly null to drop
	raw_columns = []
	for c in dataframe.columns:
		if "search_type" in c or "county_name" in c:
			raw_columns.append(c)

	# Drop previously marked null columns
	df_clean = dataframe.drop(columns=raw_columns)

	# Drop remaining rows that have nulls
	df_clean = df_clean.dropna()
	print(df_clean.count())

	return df_clean

def test_train_split(dataframe):
	# Set X for test train split and use get_dummies for one hot encoding
	X = dataframe.drop(columns=["stop_outcome"])
	X = pd.get_dummies(X)

	# Set y for test train split
	y = dataframe["stop_outcome"].copy()

	# Perform the test train split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    

if __name__ == "__main__":

	filename = "police_project.csv"

	# Converting the CSV to a panda dataframe
	dataframe = read_file(filename)

	# Data preprocessing and visualization

	# Commenting out this method since it takes care of all columns and rows
	# dataframe = general_preprocessing(dataframe)

	# Hari Analysis
	dataframe = transform_data_columns_1_4(dataframe)
	# Akash's Anaylsis
	dataframe=transform_data_columns_5_8(dataframe)
	# Jayasurya Analysis
	dataframe = transform_data_columns_9_12(dataframe)
	
	# Hari visualization
	visualize_data_columns_1_4(dataframe)
	# Akash's visualization
	visualize_data_columns_5_8(dataframe)
	# Jayasurya visualization
	visualize_data_columns_9_12(dataframe)
	# Anthony visualization
	visualize_data_columns_13_15(dataframe)

	# Perform a test train split to train our model
	# Commenting it out for now since we need to wait till all the preprocessing is done

	# X_train, X_test, y_train, y_test = test_train_split(dataframe)