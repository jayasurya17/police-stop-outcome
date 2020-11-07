import pandas as pd
import matplotlib.pyplot as plt

def read_file(filename):
	return pd.read_csv(filename)

def transform_data_columns_1_4(dataframe):
	#The county name column can be dropped cause it is completely null
	dataframe=dataframe.drop(columns=['county_name'], axis=1, inplace=False)

	# The gender dataset contains nulls. Here those nulls are discarded cause 
	# the dataset is large and the nulls consitute about 5%. Also replacing the
	# null data with frequent values might mislead the prediction
	dataframe = dataframe.dropna(subset=['driver_gender'], inplace=False)
  
	return dataframe
   
    
def visualize_data_columns_1_4(dataframe):
  
	#The statistics of the first four coumns are as follows

	#Column 1 -> Stop Date - The date on which the person is stopped

	#Column 3 -> Gender 
	print(dataframe['driver_gender'].value_counts())
	return dataframe
	
def transform_data_columns_5_8(dataframe):
	pass

def visualize_data_columns_5_8(dataframe):
	pass


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
	counts = df[column].value_counts().plot(ax=ax, kind='pie')
	plt.tight_layout()
	plt.savefig(filename)

def transform_data_columns_9_12(df): 
	# Columns
	# violation	
	# search_conducted	
	# search_type	
	# stop_outcome

	del df['search_type']
	df['stop_hour'] = df['stop_time'].apply(lambda x: int(x.split(':')[0]))
	return df

def visualize_data_columns_9_12(df):
	plot_pie_chart(df, 'search_conducted', 'search_conducted.png')
	group_and_plot_pivot_graph(df, ['stop_hour', 'violation'], 'violation.png')
	group_and_plot_pivot_graph(df, ['violation', 'stop_outcome'], 'stop_outcome.png')

def transform_data_columns_13_15(dataframe): 
	pass

def visualize_data_columns_13_15(dataframe):
	pass


if __name__ == "__main__":

	filename = "police_project.csv"

	# Converting the CSV to a panda dataframe
	dataframe = read_file(filename)

	# # Hari
	# dataframe = transform_data_columns_1_4(dataframe) 
	# dataframe = visualize_data_columns_1_4(dataframe)

	# # Akash	
	# dataframe = transform_data_columns_5_8(dataframe) 
	# dataframe = visualize_data_columns_5_8(dataframe)

	# Jayasurya
	dataframe = transform_data_columns_9_12(dataframe) 
	visualize_data_columns_9_12(dataframe)

	# Antony	
	dataframe = transform_data_columns_13_15(dataframe) 
	dataframe = visualize_data_columns_13_15(dataframe)
 