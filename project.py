import pandas as pd
from sklearn.model_selection import train_test_split


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

  filename="police_project.csv"

  # Converting the CSV to a panda dataframe
  dataframe = read_file(filename)

  # Data preprocessing and visualization

  # Commenting out this method since it takes care of all columns and rows
  # dataframe = general_preprocessing(dataframe)

  dataframe = transform_data_columns_1_4(dataframe)


  # Visualize the preprocessed dataset

  dataframe = visualize_data_columns_1_4(dataframe)


  # Perform a test train split to train our model
  # Commenting it out for now since we need to wait till all the preprocessing is done

  # X_train, X_test, y_train, y_test = test_train_split(dataframe)