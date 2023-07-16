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

def compare_means(train, cat_var, cat_value, quant_var, alt_hyp='two-sided'):
    """
    The function compares the means of two groups using the Mann-Whitney U test and returns the test
    statistic and p-value.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is a binary variable that indicates the outcome of interest. In
    this function, it is used to split the data into two groups based on the value of the target
    variable
    :param quant_var: The quantitative variable that we want to compare the means of between two groups
    :param alt_hyp: The alternative hypothesis for the Mann-Whitney U test. It specifies the direction
    of the test and can be either "two-sided" (default), "less" or "greater". "two-sided" means that the
    test is two-tailed, "less" means that the test is one, defaults to two-sided (optional)
    :return: the result of a Mann-Whitney U test comparing the means of two groups (x and y) based on a
    quantitative variable (quant_var) in a training dataset (train) with a binary target variable
    (target). The alternative hypothesis (alt_hyp) can be specified as either 'two-sided' (default),
    'less', or 'greater'.
    """
    x = train[train[quant_var]==cat_value][cat_var]
    y = train[train[quant_var]!=cat_value][cat_var]
    # alt_hyp = ‘two-sided’, ‘less’, ‘greater’
    stat,p = stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)
    print("Mann-Whitney Test:\n", f'stat = {stat}, p = {p}')

def nova(s1,s2,s3):
    '''ANOVA test for 3 samples'''
    stat,p = stats.kruskal(s1,s2,s3)
    print("Kruskal-Wallis H-Test:\n", f'stat = {stat}, p = {p}')

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

def wind_spd(train):
    '''Explore correlation of wind speed and fire size'''
    pear(train, 'wind_speed_mean', 'fire_size')
    sns.regplot(data=train,x='wind_speed_mean',y='fire_size',marker='.',line_kws={'color':'orange'})
    plt.title('Wind Speed and Wildfire Size Correlation')
    plt.xlabel('Average Wind Speed (Knots)')
    plt.ylabel('Wildfire Size (Acres)')
    plt.show()

def time(train):
    '''Explore correlation of time of day and fire size'''
    pear(train, 'time', 'fire_size')
    sns.regplot(data=train,x='time',y='fire_size',marker='.',line_kws={'color':'orange'})
    plt.title('Time of day and Wildfire Size Correlation')
    plt.xlabel('Time of day (24hr)')
    plt.ylabel('Wildfire Size (Acres)')
    plt.show()

def humid(train):
    '''Explore correlation of humidity and fire size'''
    pear(train, 'humidity_mean', 'fire_size')
    sns.regplot(data=train,x='humidity_mean',y='fire_size',marker='.',line_kws={'color':'orange'})
    plt.title('Relative Humidity and Wildfire Size Correlation')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Wildfire Size (Acres)')
    plt.show()

def temp(train):
    '''Explore correlation of temperature and fire size'''
    pear(train, 'temp_mean', 'fire_size')
    sns.regplot(data=train,x='temp_mean',y='fire_size',marker='.',line_kws={'color':'orange'})
    plt.title('Temperature and Wildfire Size Correlation')
    plt.xlabel('Average Temperature (°F)')
    plt.ylabel('Wildfire Size (Acres)')
    plt.show()

def diameter(train):
    '''Explore correlation of average tree diameter and fire size'''
    pear(train, 'diameter_mean', 'fire_size')
    sns.regplot(data=train,x='diameter_mean',y='fire_size',marker='.',line_kws={'color':'orange'})
    plt.title('Tree Diameter and Wildfire Size Correlation')
    plt.xlabel('Average Tree Diameter (Inches)')
    plt.ylabel('Wildfire Size (Acres)')
    plt.show()

def bad_human(train):
    '''Explore cause of fire and size of fire'''
    nova(train[train.cause_class=='human'].fire_size,train[train.cause_class=='natural'].fire_size,train[train.cause_class=='undetermined'].fire_size)
    plt.figure(figsize=(20,6))
    plt.subplot(121)
    sns.histplot(data=train,y='cause_class',stat='percent')
    bad_human_labels(
        'Percentage of wildfires by the determined cause',
        'Percentage of Wildfires',
    )
    plt.subplot(122)
    sns.barplot(data=train,y='cause_class',x='fire_size')
    bad_human_labels(
        'Average size of wildfires by the determined cause',
        'Average size of Wildfires (Acres)',
    )
    plt.suptitle('Are we the bad guys?')
    plt.show()

def bad_human_labels(arg0, arg1):
    '''Labels for bad human plot'''
    plt.title(arg0)
    plt.ylabel('Cause of Wildfire')
    plt.xlabel(arg1)
