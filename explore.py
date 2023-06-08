"""Explore CA Wildfire data"""

##### IMPORTS #####

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

##### FUNCTIONS #####

def pear(train, x, y, alt_hyp='two-sided'):
    '''Spearman's R test with a print'''
    r,p = stats.spearmanr(train[x], train[y], alternative=alt_hyp)
    print(f"Spearman's R: {x} and {y}\n", f'r = {r}, p = {p}')

def dist(train):
    '''Plot fire size distribution'''
    # most fires are smaller tah 5 acres (would be pretty bad otherwise)
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.hist(train[train.fire_size<10].fire_size,bins=20)
    dist_info('Wildfires less than 10 acres')
    plt.subplot(122)
    plt.hist(train[train.fire_size>=10].fire_size,bins=50)
    dist_info('Wildfires 10 acres and more')
    plt.show()

def dist_info(arg0):
    '''Sourcery suggested this so that I'm not repeating labels'''
    plt.title(arg0)
    plt.ylabel('# of Wildfires')
    plt.xlabel('Wildfire acre size')

def wind_dir(train):
    '''Explore correlation of wind direction and fire size'''
    pear(train, 'wind_direction_mean', 'fire_size')
    sns.scatterplot(data=train,x='wind_direction_mean',y='fire_size',marker='.')
    plt.show()

def time(train):
    '''Explore correlation of time of day and fire size'''
    pear(train, 'time', 'fire_size')
    sns.scatterplot(data=train,x='time',y='fire_size',marker='.')
    plt.show()

def humid(train):
    '''Explore correlation of humidity and fire size'''
    pear(train, 'humidity_mean', 'fire_size')
    sns.scatterplot(data=train,x='humidity_mean',y='fire_size',marker='.')
    plt.show()

def temp(train):
    '''Explore correlation of temperature and fire size'''
    pear(train, 'temp_mean', 'fire_size')
    sns.scatterplot(data=train,x='temp_mean',y='fire_size',marker='.')
    plt.show()

def alive(train):
    '''Explore correlation of percent living trees and fire size'''
    pear(train, 'percent_trees_alive', 'fire_size')
    sns.scatterplot(data=train,x='percent_trees_alive',y='fire_size',marker='.')
    plt.show()


