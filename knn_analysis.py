import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import common_utils as common_utils


# Fit KNN algorithm on the test and train dataset and find the accuracy with which the model was trained 
# 
# Parameters
# knn: KNN model with n neighbours that needs to be used for training
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# 
# Return: Accuracy with which training was done
def predict_and_get_accuracy(knn, X_train, X_test, y_train, y_test):
	pred = knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	result = accuracy_score(y_test, y_pred)
	return result


# Apply KNN on the entire dataset by removing one column at a time
# Find the accuracy with which model was able to predict upon removal of each column
# This shows the importance of each column in the dataset
# The number of neighbours is set to 5 while applying KNN
# 
# Parameters
# df_clean: Original dataset upon which train_test_split was applied
# 
# Return
# X_axis: Name of column that was removed
# Y_axis: Accuracy with which model was able to predict upon removal of its corresponding column
def knn_remove_columns_and_find_accuracy(df_clean):
	# Remove columns and compare
	X_axis = []
	Y_axis = []

	knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
	X_train, X_test, y_train, y_test = common_utils.test_train_split(df_clean)
	result = predict_and_get_accuracy(knn, X_train, X_test, y_train, y_test)
	X_axis.append("None")
	Y_axis.append(result)

	df_clean_orig = df_clean
	for column in df_clean_orig.columns:
		if column == "stop_outcome":
			continue
		df_clean = df_clean_orig.drop(columns=[column])
		X_train, X_test, y_train, y_test = common_utils.test_train_split(df_clean)
		result = predict_and_get_accuracy(knn, X_train, X_test, y_train, y_test)
		X_axis.append(column)
		Y_axis.append(result)
		print ("accuracy after removing column", column, ":", result)

	return X_axis, Y_axis

# Apply KNN on the entire dataset by changing the number of neighbours
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# 
# Return
# X_axis: Number of neighbours
# Y_axis: Accuracy with which model was able to predict with the mentioned number of neighbours
def knn_accuracy_on_entire_dataset(X_train, X_test, y_train, y_test):
	X_axis = []
	Y_axis = []
	for neighbors in range(2, 26):
		knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
		result = predict_and_get_accuracy(knn, X_train, X_test, y_train, y_test)
		X_axis.append(neighbors)
		Y_axis.append(result)
		print ("result for", neighbors, "neighbors (entire dataset):", result)

	return X_axis, Y_axis

# Reduce the dataset into n number of features and then apply KNN on the dataset
# Use PCA to reduce the number of features/components 
# Find accuracy of prediction using KNN with 5 neighbours 
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
# 
# Return
# X_axis: Number of features/components the dataset was reduced to
# Y_axis: Accuracy with which model was able to predict with the mentioned number of components
def apply_pca_and_compare(X_train, X_test, y_train, y_test):
	X_train_orig = X_train
	X_test_orig = X_test
	X_axis = []
	Y_axis = []
	for components in range(2, 26):
		sklearn_pca = sklearnPCA(n_components=components)
		X_test = sklearn_pca.fit_transform(X_test_orig)
		X_test = pd.DataFrame(X_test)

		X_train = sklearn_pca.fit_transform(X_train_orig)
		X_train = pd.DataFrame(X_train)

		knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
		result = predict_and_get_accuracy(knn, X_train, X_test, y_train, y_test) 
		X_axis.append(components)
		Y_axis.append(result)
		print ("result for", components, "components (after PCA):", result)

	return X_axis, Y_axis


# Apply KNN on the entire dataset by changing the number of neighbours
# Get accuracy of prediction using cross_val_score and 5 folds
# 
# Parameters
# df_clean: Original dataset upon which train_test_split was applied
# 
# Return
# X_axis: Number of neighbours
# Y_axis: Accuracy with which model was able to predict with the mentioned number of neighbours
def knn_apply_cross_val_score(df_clean):
	X = df_clean.drop(columns=["stop_outcome"])
	X = pd.get_dummies(X)
	y = df_clean["stop_outcome"].copy()
	X_axis = []
	Y_axis = []
	for neighbors in range(2, 26):
		knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
		score = cross_val_score(knn, X, y, cv = 5, scoring='accuracy')
		result = score.mean()
		X_axis.append(neighbors)
		Y_axis.append(result)
		print ("result for", neighbors, "neighbors (after cross_val_score):", result)

	return X_axis, Y_axis

# Find the accuracy with which each type of outcome was predicted using KNN
# 
# Parameters
# X_train: X_train obtained from train_test_split
# X_test: X_test obtained from train_test_split
# y_train: y_train obtained from train_test_split
# y_test: y_test obtained from train_test_split
#
# Return: accuracy with which each type of outcome was predicted using KNN
def knn_find_accuracy_of_each_class(X_train, X_test, y_train, y_test):
	for neighbors in range(5, 26, 5):
		knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
		pred = knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		# result = accuracy_score(y_test, y_pred)
		# print (result)
		results = common_utils.find_accuracy_of_each_class(y_test, y_pred)
		print ("Prediction accuracy of each class using KNN with", neighbors, "neighbors")
		print (results)


# Return a fig object that can be used to draw a plot
# 
# Parameters
# n: figure number
#
#Return: a fig object that can be used to draw a plot
def get_3d_plot(n):
	fig = plt.figure(n)
	ax = fig.add_subplot(111, projection='3d')
	return fig, ax


# Save plots with a title in a given location
# 
# Parameters
# fig: plt.figure()
# ax: subplots of the figure
# title: title that needs to be given for the figure
# location: location where the image needs to be saved
def save_plot(fig, ax, title, location):
	ax.legend()
	ax.grid(True)
	plt.title(title)
	fig.savefig(location)


# Visualize and understand how our data is clustered using scatter 3d plot
# 
# Parameters
# X_train: X_train obtained from train_test_split
# y_train: y_train obtained from train_test_split
def visualize_grouping(X_train, y_train):
	print ("visualizing data using scatter plot")
	sklearn_pca = sklearnPCA(n_components=3)
	X_train = sklearn_pca.fit_transform(X_train)
	X_train = pd.DataFrame(data=X_train, columns=["x", "y", "z"])

	df = y_train.reset_index()
	del df['index']
	df[['x', 'y', 'z']] = X_train[['x', 'y', 'z']]


	fig_combined, ax_combined = get_3d_plot(1)

	fig, ax = get_3d_plot(2)
	temp_df = df[df['stop_outcome'] == 'Arrest Driver']
	ax_combined.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'red', label = 'Arrest Driver')
	ax.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'red', label = 'Arrest Driver')
	save_plot(fig, ax, 'Police stop outcome', 'analysis_visualization/knn_clustering_arrest_driver.png')
	

	fig, ax = get_3d_plot(2)
	temp_df = df[df['stop_outcome'] == 'Arrest Passenger']
	ax_combined.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'green', label = 'Arrest Passenger')
	ax.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'green', label = 'Arrest Passenger')
	save_plot(fig, ax, 'Police stop outcome', 'analysis_visualization/knn_clustering_arrest_passenger.png')


	fig, ax = get_3d_plot(2)
	temp_df = df[df['stop_outcome'] == 'Citation']
	ax_combined.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'blue', label = 'Citation')
	ax.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'blue', label = 'Citation')
	save_plot(fig, ax, 'Police stop outcome', 'analysis_visualization/knn_clustering_citation.png')


	fig, ax = get_3d_plot(2)
	temp_df = df[df['stop_outcome'] == 'No Action']
	ax_combined.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'yellow', label = 'No Action')
	ax.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'yellow', label = 'No Action')
	save_plot(fig, ax, 'Police stop outcome', 'analysis_visualization/knn_clustering_no_action.png')



	fig, ax = get_3d_plot(2)
	temp_df = df[df['stop_outcome'] == 'Warning']
	ax_combined.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'brown', label = 'Warning')
	ax.scatter(temp_df['x'], temp_df['y'], temp_df['z'] , c= 'brown', label = 'Warning')
	save_plot(fig, ax, 'Police stop outcome', 'analysis_visualization/knn_clustering_warning.png')

	save_plot(fig_combined, ax_combined, 'Police stop outcome', 'analysis_visualization/knn_clustering_stop_outcome.png')