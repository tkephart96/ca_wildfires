# California Wildfires

![Smokey](images/smokey.jpg)

Predict the size in acres of California wildfires using data from 2001 to 2018 (due to data availability)

### Project Description

California is the most wildfire-prone state in the United States. In 2021, over 9,000 individual wildfires burned in the Southwestern state ravishing nearly 2.23 million acres. California accounted for roughly 31 percent of all acres burned due to wildfires in the US. I have decided to look into some elements that might be able to determine the size of said wildfires.

### Project Goal

* Discover drivers of wildfire size
* Use drivers to develop a machine learning model to predict wildfire size
* This can hopefully be used to better prepare people for evacuation or farmers for some early harvesting (protect the vineyards)

### Initial Thoughts

My initial hypothesis is that I will be able to use air quality and weather data (such as carbon monoxide and winds) from the Environmental Protection Agency (EPA), tree data (such as species and diameter) from the US Forest Service (USFS), and wildfire data (such as cause and location) from the US Department of Agriculture (USDA) to create a model that can roughly predict the size of wildfires in California.

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
  * Create engineered columns from existing data
* Explore data in search of drivers of wildfire size
  * Answer the following initial questions
    * What do fire do? It burns! (`placeholder`)
* Develop a Model to predict wildfire size
  * Use drivers identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Feature                   | Type            | Definition                                                                                                                                                                          |
| :------------------------ | :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| date                      | Date            | Date on which the fire was discovered or confirmed to exist and air quality measured                                                                                                |
| time                      | 24hr            | Time of day that the fire was discovered or confirmed to exist                                                                                                                      |
| cause_class               | Category        | Human or natural cause of the fire                                                                                                                                                  |
| cause                     | Category        | Specific cause of the fire                                                                                                                                                          |
| fire_size (target)        | Acres           | Estimate of acres within the final perimeter of the fire                                                                                                                            |
| fire_size_class           | Alphabetical    | Code for fire size based on the number of acres within the final fire perimeter expenditures (A=0-0.25, B=0.26-9.9, C=10.0-99.9, D=100-299, E=300 to 999, F=1000 to 4999, G=5000+) |
| lat                       | Decimal Degrees | Latitude (NAD83) for point location of the fire                                                                                                                                     |
| long                      | Decimal Degrees | Longitude (NAD83) for point location of the fire                                                                                                                                    |
| elevation_mean            | Feet            | The average distance the tree plots in the county are located above sea level                                                                                                      |
| county                    | County          | County in which the fire burned (or originated), based on nominal designation in the fire report                                                                                   |
| trees_per_acre_mean       | Numeric         | The average number of trees per acre that the tree count theoretically represents based on the sample design in the county                                                          |
| most_common_water_source  | Category        | Most common water body <1 acre in size or a stream <30 feet wide that has the greatest impact on the area within the forest land for the county                                     |
| most_common_species       | Species         | The most common tree species name in the county                                                                                                                                     |
| most_common_species_group | Genus           | The most common tree species group (Genus) name in the county                                                                                                                       |
| height_mean               | Feet            | The average total length (height) of sample trees (in feet) from the ground to the tip of the apical meristem in the county                                                        |
| diameter_mean             | Inches          | The average current diameter (in inches) of the sample trees at the point of diameter measurement in the county                                                                     |
| percent_trees_alive       | Percentage      | A percentage of the sample trees in the county are alive at the time of measurement                                                                                              |
| percent_invasive_plant    | Percentage      | A percentage of the sample trees in the county where invasive plant data was recorded                                                                                             |
| co_mean                   | PPM             | The average (arithmetic mean) value of Carbon Monoxide for the day                                                                                                                  |
| temp_mean                 | °F             | The average (arithmetic mean) value of Outdoor Temperature for the day                                                                                                              |
| humidity_mean             | %               | The average (arithmetic mean) value of Relative Humidity for the day                                                                                                                |
| wind_direction_mean       | ° Compass      | The average (arithmetic mean) value of Wind Direction for the day                                                                                                                   |
| wind_speed_mean           | Knots           | The average (arithmetic mean) value of Wind Speed for the day                                                                                                                       |
| month                     | Month           | The month of the year the fire was discovered and air quality measured                                                                                                              |
| day_of_year               | Day             | The day of the year the fire was discovered and air quality measured                                                                                                                |
| Additional Features       | Encoded         | Encoded categorical columns used for modeling                                                                                                                                       |

## Steps to Reproduce

1) Clone this repo
2) For a quick run
   - Verify `import wrangle as w` is in the imports section of final_notebook
   - Run final_notebook
   - This will use a pre-built dataset based off of the longer run datasets
3) For the longer run
   - ⚠️WARNING⚠️: These are almost the same steps I took to originally acquire the data. The steps take a lot of time and may not even be the best way of doing it. I highly recommend to do the quick run in step 2 unless you want to know how I got the data.
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
     - It will probably take awhile (millions of rows), hence I do not recommend

# Conclusions

#### Takeaways and Key Findings

* Outdoor temperature is the most correlated of the these
* While outdoor temperature, wind direction, time of day, relative humidity, and percentage of living trees are correlated, the amount is still small and may help in modeling but probably not by much
* DIfferent sized wildfires are correlated with the features differently

### Recommendations and Next Steps

* I would suggest...
* Given more time I would find a better way to more accurately merge the data, maybe even try some clustering techniques to find the 'hot zones' based on location
