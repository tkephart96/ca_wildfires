# California Wildfires

![Smokey](images/smokey.jpg)

Predict the size in acres of California wildfires using data from 2001 to 2018 (due to data availability)

### Project Description

California is the most wildfire-prone state in the United States. In 2021, over 9,000 individual wildfires burned in the Southwestern state ravishing nearly 2.23 million acres. California accounted for roughly 31 percent of all acres burned due to wildfires in the US. I have decided to look into some elements that might be able to determine the size of said wildfires.

USDA Data Citiation

- Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 [FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6

### Project Goal

* Discover drivers of wildfire size
* Use drivers to develop a machine learning model to predict wildfire size
* This can hopefully be used to better prepare people for evacuation or farmers for some early harvesting (protect the vineyards)

### Initial Thoughts

My initial hypothesis is that I will be able to use air quality and weather data (such as temperature and winds) from the Environmental Protection Agency (EPA), tree data (such as species and diameter) from the US Forest Service (USFS), and wildfire data (such as time of day and location) from the US Department of Agriculture (USDA) to create a model that can roughly predict the size of wildfires in California.

## The Plan

* Acquire data from Google BigQuery (EPA and USFS) and USDA
  * Filter by California
  * Match years so they are within the same time frame
* Prepare data
  * Pivot data (EPA)
    * Each row then represents a day with the mean of each parameter
  * Group together and get modes and means (USFS)
    * Each row then represents the most common species and its characteristic averaged
  * Merge them together based on date and relative location (county)
    * Given more time I could look into merging on lat and long
  * Clean the datasets
    * Rename columns
    * Remove nulls
    * Map categorical values
    * Handle outliers
  * Create engineered columns from existing data
    * Create month and day of year columns
    * Bin counties based on [Six Californias](https://en.wikipedia.org/wiki/Six_Californias)
    * Bin tree genus (species group) into hard or soft wood
  * Encode categorical columns
  * Split data (60/20/20)
  * Scale the data
* Explore data in search of drivers of wildfire size
  * Answer the following initial questions
    * Is there a correlation between wind speed and fire size?
    * Is there a correlation between relative humidity and fire size?
    * Is there a correlation between outdoor temperature and fire size?
    * Is there a correlation between average tree diameter and fire size?
    * Is there a difference in the the average fire size among the types of wildfire causes?
* Develop a Model to predict wildfire size
  * Use drivers identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Feature                     | Type         | Definition                                                                                                                                                                          |
| :-------------------------- | :----------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| date                        | Date         | Date on which the fire was discovered or confirmed to exist and air quality measured                                                                                                |
| time                        | 24hr         | Time of day that the fire was discovered or confirmed to exist                                                                                                                      |
| cause_class                 | Category     | Human or natural cause of the fire                                                                                                                                                  |
| cause                       | Category     | Specific cause of the fire                                                                                                                                                          |
| fire_size (target)          | Acres        | Estimate of acres within the final perimeter of the fire                                                                                                                            |
| fire_size_class             | Alphabetical | Code for fire size based on the number of acres within the final fire perimeter expenditures (A=0-0.25, B=0.26-9.9, C=10.0-99.9, D=100-299, E=300 to 999, F=1000 to 4999, G=5000+) |
| lat                         | Decimal°    | Latitude (NAD83) for point location of the fire                                                                                                                                     |
| long                        | Decimal°    | Longitude (NAD83) for point location of the fire                                                                                                                                    |
| elevation_mean              | Feet         | The average distance the tree plots in the county are located above sea level                                                                                                      |
| county                      | County       | County in which the fire burned (or originated), based on nominal designation in the fire report                                                                                   |
| trees_per_acre_mean         | Numeric      | The average number of trees per acre that the tree count theoretically represents based on the sample design in the county                                                          |
| percent_chance_water_nearby | %            | A percentage based on mean of whether a water source was nearby the sampled tree (1) or not (0) in the county                                                                       |
| most_common_species         | Species      | The most common tree species name in the county                                                                                                                                     |
| most_common_species_group   | Genus        | The most common tree species group (Genus) name in the county                                                                                                                       |
| height_mean                 | Feet         | The average total length (height) of sample trees (in feet) from the ground to the tip of the apical meristem in the county                                                        |
| diameter_mean               | Inches       | The average current diameter (in inches) of the sample trees at the point of diameter measurement in the county                                                                     |
| percent_trees_alive         | %            | A percentage of the sample trees in the county are alive at the time of measurement                                                                                              |
| percent_invasive_plant      | %            | A percentage of the sample trees in the county where invasive plant data was recorded                                                                                             |
| co_mean                     | PPM          | The average (arithmetic mean) value of Carbon Monoxide for the day in parts per millions (PPM)                                                                                      |
| temp_mean                   | °F          | The average (arithmetic mean) value of Outdoor Temperature for the day                                                                                                              |
| humidity_mean               | %            | The average (arithmetic mean) value of Relative Humidity for the day                                                                                                                |
| wind_direction_mean         | ° Compass   | The average (arithmetic mean) value of Wind Direction for the day                                                                                                                   |
| wind_speed_mean             | Knots        | The average (arithmetic mean) value of Wind Speed for the day                                                                                                                       |
| month                       | Month        | The month of the year the fire was discovered and air quality measured                                                                                                              |
| day_of_year                 | Day          | The day of the year the fire was discovered and air quality measured                                                                                                                |
| Additional Features         | Encoded      | Encoded categorical columns used for modeling                                                                                                                                       |
| six_cali                    | Categorical  | Binned counties based on the old proposal to split California into 6 states                                                                                                         |
| most_common_is_hardwood     | Boolean      | Whether or not the most common tree species in the county for that year is a hardwood tree or a softwood tree                                                                       |

## Steps to Reproduce

1) Clone this repo
   a) You may need to update your Python Libraries, my libraries were updated on 5 June, 2023 for this project
2) For a quick run
   - Verify `import wrangle as w` is in the imports section of final_notebook
   - Run final_notebook
   - This will use a pre-built and cleaned dataset based off of the datasets from the longer run in step 3
3) For the longer run
   - ⚠️WARNING⚠️: These are almost the same steps I took to originally acquire the data. The steps take a lot of time (and space) and may not even be the best way of doing it. I highly recommend to do the quick run in step 2 unless you want to know how I got the data.
   - Verify `import big_wrangle as w` is in the imports section of final_notebook
   - Install the pandas-gbq package
     - `conda install pandas-gbq --channel conda-forge`
     - `pip install pandas-gbq`
   - Go to Google BigQuery and create a project
     - Copy and run the 'long-sql' queries found in `big_wrangle.py` in [Google BigQuery](https://cloud.google.com/bigquery/public-data)
       - Click on 'Go to Datasets in Cloud Marketplace' and search for 'Historical Air Quality' or 'USFS' and view the dataset to open a quick sql prompt to query in
     - Save each result as a BigQuery table in your project
       - You can look in `big_wrangle.py` for what I named my project, database, and tables
     - Edit and save the 'small-sql' query variables found in `big_wrangle.py` to the respective table names in your BigQuery project using this format: `FROM 'database.table'` and edit the 'project_ID' variable to your project's ID
   - Run final_notebook
     - It may ask for authentication when it tries to query Google BigQuery
     - Try to run again if it stopped for authentication
   - This will run through the longer pathway of getting the datasets from the source and merge/clean/prep
     - It will probably take awhile (tens of millions of rows, +2GB), hence I do not recommend

# Conclusions

#### Takeaways and Key Findings

* Outdoor temperature is the most correlated (hot temps, dry days, hot fires)
* While outdoor temperature, wind direction, time of day, relative humidity, and percentage of living trees are correlated, the amount is still small and may help in modeling but probably not by much
* Human caused wildfires accounted for more than 70% of them and while not burning as many acres on average as nature has it still burned more acres in total
* Model was better then baseline but not significantly so, and can use improvement
  - It does predict a few acres closer the true fire sizes but not enough to make a major difference

### Recommendations and Next Steps

* I would suggest we try to get global warming under control as more and bigger fires are happening each year due to increased temperatures
* *Smokey the bear*: "Only you can prevent wildfires" 🐻👉🚫🔥🌲
* When out and about in the world, be careful when dealing with anything that can spark, give off heat, or be flammable
* Given more time I would:
  * use more air quality data for California
  * maybe not include forest data
  * handle nulls in a way other than just dropping
  * focus more on larger fires so that data is not skewed to tiny fires
  * look at the west coast states of the US or all of the states (maybe even include Canada)
  * find a better way to more accurately merge the data, maybe even try some clustering techniques to find the 'hot zones' based on location
  * take a look at the number of wildfires and sizes with a time-series approach
  * build better models, maybe even make models for each of the fire size classifications (A-G)

![Smokey](images/smokey2.jpg)
