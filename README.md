# California Wildfires

Predict the size in acres of California wildfires using data from 1992 to 2018 (wish I had the 2021 data)

### Project Description

California is the most wildfire-prone state in the United States. In 2021, over 9,000 individual wildfires burned in the Southwestern state ravishing nearly 2.23 million acres. California accounted for roughly 31 percent of all acres burned due to wildfires in the US. I have decided to look into some elements that might be able to determine the sie of wildfires.

### Project Goal

* 

### Initial Thoughts

My initial hypothesis is that I will be able to use air quality and weather data (such as carbon monoxide and winds) from the Environmental Protection Agency (EPA), tree data (such as species and diameter) from the US Forest Service (USFS), and wildfire data (such as cause and location) from the US Department of Agriculture (USDA) to create a model that can roughly predict the size of wildfires in California. This can be used to better prepare people for evacuation or farmers for some early harvesting (protect the vineyards).

## The Plan

* Acquire data from Google BigQuery (EPA and USFS) and USDA
* Prepare data
  * Clean and combine the datasets
  * Create engineered columns from existing data
* Explore data in search of drivers of wildfire size
  * Answer the following initial questions
    * 
* Develop a Model to predict wildfire size
  * Use drivers identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Original                     | Feature    | Type    | Definition                                              |
| :--------------------------- | :--------- | :------ | :------------------------------------------------------ |
|                     |        |     |               |


## Steps to Reproduce

1) Clone this repo
2) For a quick run
   - Verify `import wrangle as w` is in the imports section of final_notebook
   - Run final notebook
   - This will use a pre-built dataset based off of the longer run datasets
3) For the longer run
   - Verify `import big_wrangle as w` is in the imports section of final_notebook
   - conda or pip install `pandas-gbq` if it is not already
   - Run final notebook
   - This will run through the longer pathway of getting the datasets from the source

# Conclusions

#### Takeaways and Key Findings

* 

### Recommendations and Next Steps

* 
