'''
Wrangle CA wildfire, tree, and air quality data from a pre-built dataset

Functions:
- wrangle_wildfires
    - get_fire
    - prep_fire
- split_data
- std
'''

##### IMPORTS #####

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##### FUNCTIONS #####

def wrangle_wildfires():
    '''Wrangle together CA wildfire data'''
    # fire file
    filename = 'ca_fire.csv'
    # check for file
    if not os.path.isfile(filename):
        if not os.path.isfile('ca_fire1.csv'):
            df1 = pd.read_csv('https://media.githubusercontent.com/media/tkephart96/ca_wildfires/main/ca_fire1.csv',index_col='Unnamed: 0')
        else:
            df1 = pd.read_csv('ca_fire1.csv')
        if not os.path.isfile('ca_fire2.csv'):
            df2 = pd.read_csv('https://media.githubusercontent.com/media/tkephart96/ca_wildfires/main/ca_fire2.csv',index_col='Unnamed: 0')
        else:
            df2 = pd.read_csv('ca_fire2.csv')
        df3 = pd.concat([df1,df2])
        df3.to_csv(filename,index=False)
        return df3
    # get prebuilt wildfire date
    else:
        # read prebuilt csv
        return pd.read_csv('ca_fire.csv')

def split_data(df):
    '''Split into train, validate, test with a 60/20/20 ratio'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=42)
    return train, validate, test

def std(train,validate,test,scale=None):
    """
    The function applies the Standard Scaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    std_scale = StandardScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(std_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(std_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(std_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt