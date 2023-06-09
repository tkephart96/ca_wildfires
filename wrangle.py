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
        # pull prebuilt from gist.github
        df = pd.read_csv('https://gist.githubusercontent.com/tkephart96/21d138cad542f0a8a123ba02911613c1/raw/f87b42bc585be00779d8e56a30b45787d85691ff/ca_fire.csv')
        # cache it
        df.to_csv('ca_fire.csv',index=False)
        return df
    # get prebuilt wildfire date
    else:
        # read prebuilt csv
        return pd.read_csv('ca_fire.csv')

def encode(df):
    '''Encode categorical columns'''
    # columns to encode
    cols = ['cause_class','six_cali']
    # cols = ['cause_class','cause','county','most_common_water_source','most_common_species','most_common_species_group']
    # encode the dummies
    dummy = pd.get_dummies(df[cols])
    # bring the dummies along
    return pd.concat([df,dummy],axis=1)

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
    Xtr = pd.DataFrame(std_scale.fit_transform(train[scale]),train[scale].index,train[scale].columns)
    Xv = pd.DataFrame(std_scale.transform(validate[scale]),validate[scale].index,validate[scale].columns)
    Xt = pd.DataFrame(std_scale.transform(test[scale]),test[scale].index,test[scale].columns)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt
