import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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
	X_train, X_test, y_train, y_test = test_train_split(df_clean)
	result = predict_and_get_accuracy(knn, X_train, X_test, y_train, y_test)
	X_axis.append("None")
	Y_axis.append(result)

	df_clean_orig = df_clean
	for column in df_clean_orig.columns:
		if column == "stop_outcome":
			continue
		df_clean = df_clean_orig.drop(columns=[column])
		X_train, X_test, y_train, y_test = test_train_split(df_clean)
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
	for neighbors in range(2, 36):
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
	for components in range(2, 36):
		sklearn_pca = sklearnPCA(n_components=components)
		X_test = sklearn_pca.fit_transform(X_test_orig)
		X_test = pd.DataFrame(X_test)

		X_train = sklearn_pca.fit_transform(X_train_orig)
		X_train = pd.DataFrame(X_train)

		knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
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
	for neighbors in range(2, 36):
		knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
		score = cross_val_score(knn, X, y, cv = 5, scoring='accuracy')
		result = score.mean()
		X_axis.append(neighbors)
		Y_axis.append(result)
		print ("result for", neighbors, "neighbors (after cross_val_score):", result)

	return X_axis, Y_axis


def get_graph():
	fig, ax = plt.subplots()
	plt.plot(range(2, 31), 
	[0.9090538336052202, 0.8994406898158938, 0.9086460032626428, 0.9042763924493125, 0.9079468655325099, 0.905558144954556, 0.9067233745047775, 0.905499883477045, 0.9057911908646004, 0.9050920531344675, 0.905499883477045, 0.9048590072244233, 0.90468422279189, 0.9040433465392682, 0.9040433465392682, 0.9035772547191797, 0.9034607317641575, 0.9027615940340247, 0.9025285481239804, 0.9023537636914473, 0.9021207177814029, 0.9015963644838033, 0.9017711489163365, 0.9012467956187369, 0.9010720111862037, 0.900547657888604, 0.9008389652761594, 0.900547657888604, 0.9003146119785598, 0.9000233045910044, 0.9002563505010487, 0.9000233045910044, 0.9000815660685155, 0.8998485201584712, 0.8994989512934048, 0.8994406898158938, 0.8993241668608716, 0.8987415520857609, 0.8988580750407831, 0.8987415520857609, 0.8986832906082498, 0.8983919832206945, 0.8981589373106502, 0.8982754602656723, 0.8982754602656723, 0.897984152878117, 0.898042414355628, 0.8978093684455838, 0.8976928454905616]
	, label='KNN accuracy for n neighbors on entire dataset (82 features)', c="blue")
	plt.plot(range(2, 31), 
	[0.6000932183640177, 0.7787229084129573, 0.8856909811232813, 0.8885457935213237, 0.8825448613376835, 0.8833605220228385, 0.8810300629223957, 0.8844674900955488, 0.8848170589606152, 0.8864483803309252, 0.8882544861337683, 0.8889536238639012, 0.8880797017012352, 0.8874388254486134, 0.8867979491959916, 0.8892449312514565, 0.888720577953857, 0.8873223024935912, 0.8881379631787462, 0.8867979491959916, 0.887205779538569, 0.8877301328361688, 0.8892449312514565, 0.8889536238639012, 0.8876136098811466, 0.8870309951060359, 0.8884875320438126, 0.8888953623863901, 0.8998485201584712, 0.8983337217431834, 0.8999067816359823, 0.8995572127709158, 0.8991493824283384, 0.8997902586809602, 0.8992659053833605, 0.8997319972034491, 0.9003146119785598, 0.8994989512934048, 0.8991493824283384, 0.8992076439058495, 0.8989163365182941, 0.8992076439058495, 0.8981589373106502, 0.8988580750407831, 0.8998485201584712, 0.9000815660685155, 0.8989745979958051, 0.8991493824283384, 0.8993241668608716]
	, label='KNN accuracy after applying PCA and reducing to n components', c="red")
	plt.plot(range(2, 31), 
	[0.86772711648944, 0.8294348266933245, 0.8745091963596476, 0.8566798345178234, 0.882048839875636, 0.8715376813602946, 0.8871296903055772, 0.8800795287465333, 0.8899731260558615, 0.8854517036732, 0.8931661544339489, 0.890194600463708, 0.8949025109019156, 0.8923970915246893, 0.8963009441467522, 0.8942965915358007, 0.8972215828989965, 0.8956950076714669, 0.8980722755609056, 0.8971866450226792, 0.8990861288751297, 0.8984452127011104, 0.8991444245709814, 0.8987831786971668, 0.8995290035510842, 0.8992143782653922, 0.8999718478107364, 0.8995989505919287, 0.9002631922681607, 0.899727156258756, 0.900426353870985, 0.8999835400276358, 0.9000884382519302, 0.9001700081224835, 0.9003215031721636, 0.9000534899200089, 0.900473008677448, 0.9002399561138377, 0.9005079750690493, 0.9004031006074916, 0.9005312910661674, 0.9001700508954092, 0.9004264261097042, 0.9002050153859917, 0.9004264384663271, 0.9002283389871854, 0.9002399912826876, 0.9001584299667194, 0.9001118217352201]
	, label='cross_val_score with 5 fold for n neighbors', c="green")
	plt.xlabel("n")
	plt.ylabel("score")
	ax.legend()
	plt.tight_layout()
	plt.savefig("KNN.png")