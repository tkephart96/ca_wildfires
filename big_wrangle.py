'''
Wrangle the data the same way I did

BigQuery Stuff:
- long_sql_air_quality
- long_sql_usfs
- small_sql_air_quality
- small_sql_usfs
- project_ID

Functions:
- wrangle_wildfires
    - wrangle_forest_fires
        - wrangle_fires
            - get_zipped_fire
            - get_fires
            - prep_fires
        - wrangle_forest
            - get_forest
            - prep_forest
    - wrangle_air_quality
        - get_air_quality
        - prep_air_quality
- split_data
- std
'''

##### IMPORTS #####

import numpy as np
import pandas as pd
import sqlite3
import requests
import zipfile
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##### BIGQUERY STUFF #####

# state_code 6 is CA
# datum can't be unknown or NAD27 as I want lat long to be somewhat accurate
# the 2.5ft error between NAD83 and WGS84 shouldn't affect this project (I think)
# this in turn leaves me with data from 1997 but usfs only has from 2001
long_sql_air_quality = """
SELECT poc,latitude,longitude,parameter_name,sample_duration,
date_local,units_of_measure,observation_count,
arithmetic_mean,county_code,site_num
FROM `bigquery-public-data.epa_historical_air_quality.co_daily_summary`
WHERE state_code = '06' and date_local < '2019-01-01' and date_local > '2000-12-31' and datum != 'UNKNOWN' and datum != 'NAD27'
UNION ALL
SELECT poc,latitude,longitude,parameter_name,sample_duration,
date_local,units_of_measure,observation_count,
arithmetic_mean,county_code,site_num
FROM `bigquery-public-data.epa_historical_air_quality.pressure_daily_summary`
WHERE state_code = '06' and date_local < '2019-01-01' and date_local > '2000-12-31' and datum != 'UNKNOWN' and datum != 'NAD27'
UNION ALL
SELECT poc,latitude,longitude,parameter_name,sample_duration,
date_local,units_of_measure,observation_count,
arithmetic_mean,county_code,site_num
FROM `bigquery-public-data.epa_historical_air_quality.rh_and_dp_daily_summary`
WHERE state_code = '06' and date_local < '2019-01-01' and date_local > '2000-12-31' and datum != 'UNKNOWN' and datum != 'NAD27'
UNION ALL
SELECT poc,latitude,longitude,parameter_name,sample_duration,
date_local,units_of_measure,observation_count,
arithmetic_mean,county_code,site_num
FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
WHERE state_code = '06' and date_local < '2019-01-01' and date_local > '2000-12-31' and datum != 'UNKNOWN' and datum != 'NAD27'
UNION ALL
SELECT poc,latitude,longitude,parameter_name,sample_duration,
date_local,units_of_measure,observation_count,
arithmetic_mean,county_code,site_num
FROM `bigquery-public-data.epa_historical_air_quality.wind_daily_summary`
WHERE state_code = '06' and date_local < '2019-01-01' and date_local > '2000-12-31' and datum != 'UNKNOWN' and datum != 'NAD27'
"""

# usfs CA data is only from 2001 to 2018 (this is where I am limited most)
long_sql_usfs = """
SELECT  
    measurement_year,
    tree_county_code,latitude,longitude,elevation,
    trees_per_acre_unadjusted,
    water_on_plot_code_name,
    species_common_name,species_group_code_name,
    total_height,
    current_diameter,
    tree_status_code_name,
    invasive_sampling_status_code_name,
FROM `bigquery-public-data.usfs_fia.plot_tree`
WHERE plot_state_code = 6 AND total_height > 0 AND measurement_year > 2000
"""

# EDIT THE BELOW 'FROM' lines in both small_sql variables
# with your respective database and table names from your project
# my database is `california_air_trees_fires`
# my air quality table is `epa_aq_ca5`
small_sql_air_quality = """
SELECT *
FROM `california_air_trees_fires.epa_aq_ca5`
"""

# my usfs table is `usfs_fia_pt_ca2`
small_sql_usfs = """
SELECT *
FROM `california_air_trees_fires.usfs_fia_pt_ca2`
"""

# EDIT THE BELOW project_ID with your respective project ID
project_ID = 'my-ds-projects'

##### FUNCTIONS #####

def get_zipped_fire():
    '''Get the necessary sqlite file from the zip without having to save the whole zip'''
    # url for zipped data
    url = 'https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.6/RDS-2013-0009.6_SQLITE.zip'
    # get zipped data
    response = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    # extract necessary file
    z.extract('Data/FPA_FOD_20221014.sqlite')

def get_fires():
    '''Get fire data from sqlite file and download file if not there'''
    # sqlite connection to file downloaded from zip
    filename = 'Data/FPA_FOD_20221014.sqlite'
    if not os.path.isfile(filename):
        # caches sqlite file
        get_zipped_fire()
    # make connection to sqlite file
    conn = sqlite3.connect(filename)
    # read sql query
    return pd.read_sql(
                '''
                        select 
                        FIRE_YEAR,
                        DISCOVERY_DATE,
                        DISCOVERY_TIME,
                        NWCG_CAUSE_CLASSIFICATION,
                        NWCG_GENERAL_CAUSE,
                        FIRE_SIZE,
                        FIRE_SIZE_CLASS,
                        LATITUDE,
                        LONGITUDE,
                        FIPS_CODE,
                        FIPS_NAME
                        from fires where STATE = :state
                    ''',
        conn,
        params={'state': 'CA'},)

def prep_fires(fire):
    '''Prepare fire data'''
    # make the columns stop yelling at me
    fire.columns = fire.columns.str.lower()
    # get rid of a few nulls to create county_code column
    fire = fire[fire['fips_code'].notna()].copy()
    # need for merge
    fire['county_code'] = fire.fips_code.str[2:].astype(int)
    return fire

def wrangle_fires():
    '''Wrangle some fire'''
    # get fire data
    fire = get_fires()
    # prep fire data and put it out
    return prep_fires(fire)

def get_air_quality():
    '''Get the saved air quality data from personal project on BigQuery'''
    # pandas gbq will auth with google account first, rerun again if necessary
    return pd.read_gbq(small_sql_air_quality,dialect='standard',project_id=project_ID,use_bqstorage_api=True)

def prep_air_quality(aq):
    '''Prepare air quality data'''
    # change to datetime, need for pivot and merge
    aq.date_local = aq.date_local.astype('datetime64[ns]')
    # create year column, need for concat
    aq['year'] = aq.date_local.dt.year
    # convert county_code to int, need for pivot and merge
    aq.county_code = aq.county_code.astype(int)
    return aq

def wrangle_air_quality():
    '''Wrangle some air'''
    # get air quality
    aq = get_air_quality()
    # prep air data
    aq = prep_air_quality(aq)
    # set blank variable for fully pivoted data to concat on
    caq = None
    # loop thru years for easy pivot and concat
    for year in range(2001,2019):
        # air quality per year
        sm_aq = aq[(aq.year==year)]
        # pivot so row is each county on date with mean of each parameter
        piv_aq = sm_aq.pivot_table(index=['date_local','county_code'],
                                    columns='parameter_name',
                                    values=['arithmetic_mean']).reset_index()
        # concat together
        caq = piv_aq if caq is None else pd.concat([caq,piv_aq],ignore_index=True)
    # cough up air quality data
    return caq

def get_forest():
    '''Get the saved forest data from personal project on BigQuery'''
    # pandas gbq will auth with google account first, rerun again if necessary
    return pd.read_gbq(small_sql_usfs,dialect='standard',project_id=project_ID,use_bqstorage_api=True)   

def prep_forest(pt):
    '''Prepare forest data'''
    # I use pt because usfs table was called plot_tree
    # drop them na trees
    pt = pt[pt['tree_county_code'].notna()].copy()
    # map easy categorical values to 1 and 0
    pt.tree_status_code_name = pt.tree_status_code_name.map({'Live tree':1,'Dead tree':0})
    pt.invasive_sampling_status_code_name = pt.invasive_sampling_status_code_name.map({'Invasive plant data collected on all accessible land conditions':1,'Not collecting invasive plant data':0})
    # fix dtype
    pt.water_on_plot_code_name = pt.water_on_plot_code_name.astype(str)
    # boolean water values
    pt.water_on_plot_code_name = pt.water_on_plot_code_name.map({
        'Flood zones - evidence of flooding WHEN water_on_plot_code = bodies of water exceed their natural banks':1,
        'None - no water sources within the accessible forest land condition class':0,
        'Other temporary water':1,
        'Permanent streams or ponds too small to qualify as noncensus water':1,
        'Temporary streams':1,
        '':0,
        'Ditch/canal - human-made channels used as a means of moving water, e.g., for irrigation or drainage, which are too small to qualify as noncensus water':1,
        'Permanent water in the form of deep swamps, bogs, marshes without standing trees present and less than 1.0 acre in size, or with standing trees':1})
    # Performed 9 aggregations grouped on columns: 'measurement_year', 'tree_county_code'
    # used for merge so each row has most common tree and stats for county and year
    return pt.groupby(
        ['measurement_year', 'tree_county_code']
            ).agg(
                elevation_mean=('elevation', 'mean'), 
                trees_per_acre_mean=('trees_per_acre_unadjusted', 'mean'),
                percent_chance_water_nearby=('water_on_plot_code_name', 'mean'),
                most_common_species=('species_common_name', lambda s: s.value_counts().index[0]),
                most_common_species_group=('species_group_code_name', lambda s: s.value_counts().index[0]),
                height_mean=('total_height', 'mean'),
                diameter_mean=('current_diameter', 'mean'),
                percent_trees_alive=('tree_status_code_name', 'mean'),
                percent_invasive_plant=('invasive_sampling_status_code_name', 'mean')
                ).reset_index()

def wrangle_forest():
    '''Wrangle some trees'''
    # get tree data
    pt = get_forest()
    # prep trees and roll them out
    return prep_forest(pt)

def wrangle_forest_fires():
    # get fire and forest data for merge
    fire = wrangle_fires()
    pt = wrangle_forest()
    # set blank variable for fully merged data to concat on
    fpt = None
    # loop thru years for easy merge and concat
    for year in range(2001,2019):
        # fire each year
        sm_fire = fire[fire.fire_year==year]
        # tree each year
        sm_pt = pt[pt.measurement_year==year]
        # dropping tree nulls for merge
        sm_pt = sm_pt.dropna()
        # fire tree merge on county to fires
        sm_fpt = pd.merge(left=sm_fire,right=sm_pt,how='left',
                            left_on='county_code',right_on='tree_county_code')
        # concat together
        fpt = sm_fpt if fpt is None else pd.concat([fpt,sm_fpt],ignore_index=True)
    # convert to datetime for merge with air quality
    fpt.discovery_date = fpt.discovery_date.astype('datetime64[ns]')
    # put out the forest fires
    return fpt

def wrangle_wildfires():
    '''Wrangle together CA wildfire data'''
    # fire file
    filename = 'ca_fire.csv'
    # check for file
    if not os.path.isfile(filename):
        # get forest fire and air data for merge
        fpt = wrangle_forest_fires()
        caq = wrangle_air_quality()
        # forest fire and air data merge on data and county to forest fire
        ca = pd.merge(left=fpt,right=caq,how='left',
                        left_on=['discovery_date','county_code'],
                        right_on=[('date_local',''),('county_code','')])
        # rename for readability and ease of use
        ca.columns = [
            'fire_year',
            'date',
            'time',
            'cause_class',
            'cause',
            'fire_size',
            'fire_size_class',
            'lat',
            'long',
            'fips_code',
            'county',
            'county_code1',
            'measurement_year',
            'tree_county_code',
            'elevation_mean',
            'trees_per_acre_mean',
            'percent_chance_water_nearby',
            'most_common_species',
            'most_common_species_group',
            'height_mean',
            'diameter_mean',
            'percent_trees_alive',
            'percent_invasive_plant',
            'date_local',
            'county_code2',
            'bp_mean',
            'co_mean',
            'dp_mean',
            'temp_mean',
            'humidity_mean',
            'wind_direction_mean',
            'wind_speed_mean']
        # keep necessary columns, drop others
        ca = ca[[
            'date',
            'time',
            'cause_class',
            'cause',
            'fire_size',
            'fire_size_class',
            'lat',
            'long',
            'elevation_mean',
            'county',
            'trees_per_acre_mean',
            'percent_chance_water_nearby',
            'most_common_species_group',
            'height_mean',
            'diameter_mean',
            'percent_trees_alive',
            'percent_invasive_plant',
            'co_mean',
            'temp_mean',
            'humidity_mean',
            'wind_direction_mean',
            'wind_speed_mean']]
        # get rid of nulls
        ca = ca.dropna()
        # make datetime to create features
        ca.date = ca.date.astype('datetime64[ns]')
        # make all values lowercase
        for col in ca.select_dtypes(include=('object')).columns:
            ca[col] = ca[col].str.lower()
        # make time features
        ca['month'] = ca.date.dt.month.copy()
        ca['day_of_year'] = ca.date.dt.dayofyear.copy()
        # bin counties
        ca.county = ca.county.str.replace(' county','')
        jefferson = ['butte', 'colusa', 'del norte', 'glenn', 'humboldt', 'lake', 'lassen', 'mendocino', 'modoc', 'plumas', 'shasta', 'siskiyou', 'tehama', 'trinity']
        north_cali = ['amador', 'el dorado', 'marin', 'napa', 'nevada', 'placer', 'sacramento', 'sierra', 'solano', 'sonoma', 'sutter', 'yolo', 'yuba']
        silicon_valley = ['alameda', 'contra costa', 'monterey', 'san benito', 'san francisco', 'san mateo', 'santa clara', 'santa cruz']
        central_cali = ['alpine', 'calaveras', 'fresno', 'inyo', 'kern', 'kings', 'madera', 'mariposa', 'merced', 'mono', 'san joaquin', 'stanislaus', 'tulare', 'tuolumne']
        west_cali = ['los angeles', 'san luis obispo', 'santa barbara', 'ventura']
        south_cali = ['imperial', 'orange', 'riverside', 'san bernardino', 'san diego']
        ca['six_cali'] = np.where(ca.county.isin(jefferson),'jefferson','')
        ca['six_cali'] = np.where(ca.county.isin(north_cali),'north_cali',ca['six_cali'])
        ca['six_cali'] = np.where(ca.county.isin(silicon_valley),'silicon_valley',ca['six_cali'])
        ca['six_cali'] = np.where(ca.county.isin(central_cali),'central_cali',ca['six_cali'])
        ca['six_cali'] = np.where(ca.county.isin(west_cali),'west_cali',ca['six_cali'])
        ca['six_cali'] = np.where(ca.county.isin(south_cali),'south_cali',ca['six_cali'])
        # bin species group hard or soft
        softwoods = ['woodland softwoods','other western softwoods','douglas-fir','lodgepole pine','ponderosa and jeffrey pines','true fir','redwood','incense-cedar']
        hardwoods = ['woodland hardwoods','other western hardwoods','cottonwood and aspen','oak']
        ca['most_common_is_hardwood'] = np.where(ca.most_common_species_group.isin(hardwoods),1,0)
        # map cause class values
        ca.cause_class = ca.cause_class.map({
            'human':'human',
            'natural':'natural',
            'missing data/not specified/undetermined':'undetermined'
        })
        # handle outlier fire size, if it's 1000 acres that's bad
        ca = ca[ca['fire_size'] < 1000]
        # cache locally
        ca.to_csv('ca_fire.csv',index=False)
        # put out wildfires
        return ca
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