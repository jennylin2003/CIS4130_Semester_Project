#!/usr/bin/env python
# coding: utf-8

# # Milestone 3: Descriptive Statistics

# The data exploration process included getting basic statistics such as list of variables, missing values per column, and imputation of variables with high null value counts. Also created some visualizations to analyze relationships between variables.

# In[1]:


from pyspark.sql import SparkSession


# In[2]:


get_ipython().system('hdfs dfsadmin -safemode leave')


# In[3]:


from pyspark.sql import SparkSession


# In[4]:


# Create a Spark session with increased executor memory
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()


# In[5]:


df = spark.read.csv("gs://my-bigdatatech-project-jl/landing/itineraries.csv", header=True, inferSchema=True)


# In[6]:


print(f"Number of records: {df.count()}")


# In[7]:


# List of variables
variables = df.columns
print(variables)


# In[8]:


from pyspark.sql import functions as F

# Number of missing values per column
missing_values = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])

# Show result
missing_values.show()


# In[9]:


from pyspark.sql.functions import col, when

def count_nulls(df, segmentsEquipmentDescription):
    """
    Function to count the number of null values in a given column of a DataFrame.
    
    :param df: The PySpark DataFrame
    :param column_name: The column for which to count null values
    :return: The count of null values in the specified column
    """
    return df.select(when(col(segmentsEquipmentDescription).isNull(), 1).alias(segmentsEquipmentDescription)).groupBy().sum(segmentsEquipmentDescription).collect()[0][0]

# Example usage
null_count = count_nulls(df, "segmentsEquipmentDescription")
print(f"Number of null values in segmentsEquipmentDescription: {null_count}")


# In[10]:


def count_nulls(df, totalTravelDistance):
    """
    Function to count the number of null values in a given column of a DataFrame.
    
    :param df: The PySpark DataFrame
    :param column_name: The column for which to count null values
    :return: The count of null values in the specified column
    """
    return df.select(when(col(totalTravelDistance).isNull(), 1).alias(totalTravelDistance)).groupBy().sum(totalTravelDistance).collect()[0][0]

null_count = count_nulls(df, "totalTravelDistance")
print(f"Number of null values in totalTravelDistance: {null_count}")


# In[11]:


# Get the first 10 rows of the segmentsEquipmentDescription column
df.select("segmentsEquipmentDescription").show(10)


# In[12]:


# Get statistics with df.describe() function
# find min, max, avg, std dev for all numeric variables
numeric_stats = df.describe() 
numeric_stats.show()


# In[13]:


from pyspark.sql import functions as F

# get min and max dates for date variables
columns = df.columns

date_columns = [c for c in columns if 'date' in c.lower()]

for date_col in date_columns:
    min_date = df.select(F.min(date_col)).first()[0]
    max_date = df.select(F.max(date_col)).first()[0]
    print(f"{date_col} - Min date: {min_date}, Max date: {max_date}")


# In[14]:


# Filter the Spark DataFrame (example condition)
filtered_df = df.filter(df.totalFare.isNotNull() & df.segmentsEquipmentDescription.isNotNull())

# Limit to 10 rows
limited_df = filtered_df.limit(50)

# Convert to Pandas DataFrame
pandas_df = limited_df.toPandas()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Create a boxplot
sns.boxplot(x='segmentsEquipmentDescription', y='totalFare', data=pandas_df)

# Set title and labels
plt.title('Boxplot of Total Fare vs. Equipment Description')
plt.xlabel('Segments Equipment Description')
plt.ylabel('Total Fare')

plt.yticks(fontsize=10)  
plt.xticks(fontsize=10)

# Rotate x-axis labels for better readability (if needed)
plt.xticks(rotation=90)

# Show the plot
plt.tight_layout()
plt.show()


# In[16]:


# get distribution of total fare for one-way flights
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['totalFare'], bins=30, kde=True)
plt.title('distribution of fare for One-Way Flights')
plt.xlabel('total fare (USD)')
plt.ylabel('frequency')
plt.grid()
plt.show()


# In[17]:


# scatterplot to see if seats remaining and total fare have a strong correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='seatsRemaining', y='totalFare', data=pandas_df)
plt.title('Seats Remaining vs. Total Fare')
plt.xlabel('Seats Remaining')
plt.ylabel('Total Fare (USD)')
plt.grid()
plt.show()


# In[18]:


# travel distance vs total fare
plt.figure(figsize=(12, 6))
sns.boxplot(x='totalTravelDistance', y='totalFare', data=pandas_df)
plt.title('Total Fare vs. Travel Distance')
plt.xlabel('Total Travel Distance (Miles)')
plt.ylabel('Total Fare (USD)')
plt.grid()
plt.show()


# In[19]:


# travel duration vs total fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='travelDuration', y='totalFare', data=pandas_df)
plt.title('Relationship Between Travel Duration and Total Fare')
plt.xlabel('Travel Duration (HH:MM)')
plt.ylabel('Total Fare (USD)')
plt.grid()
plt.show()


# In[20]:


# ticket price over time
import pandas as pd

pandas_df['searchDate'] = pd.to_datetime(pandas_df['searchDate'])
average_fare_over_time = pandas_df.groupby('searchDate')['totalFare'].mean()

plt.figure(figsize=(12, 6))
plt.plot(average_fare_over_time.index, average_fare_over_time, marker='o')
plt.title('Average Total Fare Over Time')
plt.xlabel('Date')
plt.ylabel('Average Total Fare (USD)')
plt.grid()
plt.xticks(rotation=45)
plt.show()


# In[21]:


# fare comparison of non-stop vs connecting flights
plt.figure(figsize=(10, 6))
sns.violinplot(x='isNonStop', y='totalFare', data=pandas_df)
plt.title('Total Fare: Non-Stop vs. Connecting Flights')
plt.xlabel('Non-Stop Flight')
plt.ylabel('Total Fare (USD)')
plt.xticks([0, 1], ['Connecting', 'Non-Stop'])
plt.grid()
plt.show()


# In[22]:


# analyze fare variations between diff routes
plt.figure(figsize=(12, 8))
sns.boxplot(x='startingAirport', y='totalFare', data=pandas_df)
plt.title('Total Fare by Starting Airport')
plt.xlabel('Starting Airport')
plt.ylabel('Total Fare (USD)')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# In[23]:


# average total fare by airline
average_fare_by_airline = pandas_df.groupby('segmentsAirlineName')['totalFare'].mean().sort_values()
plt.figure(figsize=(10, 6))
average_fare_by_airline.plot(kind='barh')
plt.title('Average Total Fare by Airline')
plt.xlabel('Average Total Fare (USD)')
plt.ylabel('Airline')
plt.grid()
plt.show()


# In[24]:


# use a histogram to analyze how ticket prices are distributed across all flights
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['totalFare'], bins=30, kde=True)
plt.title('Distribution of Total Fare for One-Way Flights')
plt.xlabel('Total Fare (USD)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[25]:


# see if there's a correlation between num of seats remaining and total fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='seatsRemaining', y='totalFare', data=pandas_df)
plt.title('Seats Remaining vs. Total Fare')
plt.xlabel('Seats Remaining')
plt.ylabel('Total Fare (USD)')
plt.grid()
plt.show()


# In[26]:


get_ipython().system('jupyter nbconvert --to pdf EDA.ipynb')


# In[27]:


# save the EDA to a Parquet file
EDA_data_path = "gs://my-bigdatatech-project-jl/landing/EDA.ipynb"
df.write.parquet(EDA_data_path)


# In[ ]:




