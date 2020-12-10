import pandas as pd
from sklearn.model_selection import train_test_split


def test_train_split(dataframe):
	# Set X for test train split and use get_dummies for one hot encoding
	X = dataframe.drop(columns=["stop_outcome"])
	X = pd.get_dummies(X)

	# Set y for test train split
	y = dataframe["stop_outcome"].copy()

	# Perform the test train split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	return X_train, X_test, y_train, y_test


# Find the accuracy with which each type of outcome was predicted
# 
# Parameters
# y_test: Test data obtained from test_train_split
# y_pred: Predicted values by an algorithm. This is of type list
# Number of rows in y_test must be equal to length of list y_pred
#
# Return: accuracy with which each type of outcome was predicted
def find_accuracy_of_each_class(y_test, y_pred):
	new_df = y_test.reset_index()
	del new_df['index']

	new_df['pred'] = y_pred
	new_df['result'] = new_df.apply(lambda x: x['stop_outcome'] == x['pred'], axis = 1)

	pred_results = new_df.groupby(['stop_outcome', 'result']).size().reset_index()
	pred_count = new_df.groupby(['stop_outcome']).size().reset_index()

	pred_results = pred_results[pred_results.result == True]
	pred_results = pd.merge(pred_results, pred_count, on='stop_outcome', how='right')
	pred_results['accuracy'] = pred_results.apply(lambda x: x['0_x'] / x['0_y'], axis = 1)
	# print (pred_results)
	
	del pred_results['result']
	del pred_results['0_x']
	del pred_results['0_y']
	pred_results['accuracy'] = pred_results['accuracy'].fillna(value=0)

	return pred_results