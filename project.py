import pandas as pd



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


    



   
    
if __name__ == "__main__":

  filename="police_project.csv"

  # Converting the CSV to a panda dataframe
  dataframe = read_file(filename)

  # Data cleaning , fix missing
  dataframe = transform_data_columns_1_4(dataframe) 

  dataframe = visualize_data_columns_1_4(dataframe)
 


     
    

    


