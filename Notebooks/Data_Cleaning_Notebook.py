#!/usr/bin/env python
# coding: utf-8

# # CS 3500 - Starter Notebook 📒

# In[74]:


# Check current version of Jupyter
get_ipython().system('jupyter --version')


# In[75]:


# Here several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from datetime import datetime, timedelta


# In[76]:


#importing libraries
import numpy as np
import regex
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# ## Read Data Set and Look at  Metadata

# In[77]:


#reading the CSV file into dataframe df
# Data should be located in the same folder as the notebook for this to work
df = pd.read_csv('../Data/Dirty/LA_Crime_Data_2023_to_Present_data.csv') 


# In[78]:


# print shape of dataframe
print(df.shape)


# In[79]:


# print basic info about the data frame
df.info()


# In[80]:


# print first 5 records of the dataframe
df.head()


# In[81]:


# Set column 'DR_NO' Index
df = df.set_index('DR_NO')
df.info()


# In[82]:


# Drop uneccesary columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Checking dataframe
df.info()


# In[83]:


# Convert the columns to a suitable data type
df['Date Rptd'] = df['Date Rptd'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
df['DATE OCC'] = df['DATE OCC'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df['AREA NAME'] = df['AREA NAME'].astype('string')
df['Crm Cd Desc'] = df['Crm Cd Desc'].astype('string')
df['Mocodes'] = df['Mocodes'].astype('string')
df['Vict Sex'] = df['Vict Sex'].astype('string')
df['Vict Descent'] = df['Vict Descent'].astype('string')
df['Premis Desc'] = df['Premis Desc'].astype('string')
df['Weapon Desc'] = df['Weapon Desc'].astype('string')
df['Status'] = df['Status'].astype('string')
df['Status Desc'] = df['Status Desc'].astype('string')

# Checking dataframe
df.info()


# In[84]:


# Map Target Column

# Mapping dictionary
mapping = {
            'IC': 'No Arrest'
            ,'AA': 'Arrest'
            ,'AO': 'No Arrest'
            ,'JO': 'No Arrest'
            ,'JA': 'Arrest'
            ,'CC': 'No Arrest'
}
# Create target variable based in the status variable 
df['Target'] = df['Status'].map(mapping)

# Checking dataframe
df.info()


# In[85]:


# Count values in 'col1'
value_counts = df['Target'].value_counts()
print(value_counts)


# ##### Look at "TIME OCC" data

# In[86]:


# Look at time data
df.loc[: ,'TIME OCC'].head(10)


# TIME OCC column is not in the right format and there are few Discripencies(Like 30, 40 as shown below and inserting ':' to correctly represent in the format HH:MM )
# 
# Changing 30 to 00:30
# Correcting the Format of the ones that are meaningful - 1200 to 12:00

# In[87]:


# Change data type
df['TIME OCC'] = df['TIME OCC'].astype('string')

# Pad the 'TIME OCC' column values with leading zeros to ensure a 4-digit format
df['TIME OCC'] = df['TIME OCC'].str.zfill(4)

# Format the 'TIME OCC' column as 'HH:MM' (hour:minute)
df['TIME OCC'] = df['TIME OCC'].str[:-2] + ':' + df['TIME OCC'].str[-2:]


# In[88]:


# Change data type
df['TIME OCC'] = df['TIME OCC'].astype('string')

# Print the firt 10 rows of the column
df.loc[: ,'TIME OCC'].head(10)


# In[89]:


# write data to see dataset
df.to_csv('df.csv', index=False)


# ## Classify Data 🔎 

# In[90]:


# List columns
df.columns


# In[91]:


# Select some features, you can add or take out some.
col_target = ['Target']
cols_numerical = ['AREA', 'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Vict Age', 'Premis Cd', 'Weapon Used Cd'] 
cols_categorical = ['TIME OCC', 'AREA NAME', 'Crm Cd Desc', 'Mocodes', 'Vict Sex', 'Vict Descent', 'Premis Desc', 'Weapon Desc']
cols_datetime = ['Date Rptd', 'DATE OCC']

#Dropping AREA as both 'AREA' and 'AREA NAME' represent the same.
cols_other = ['Status', 'Status Desc']


# ## Check for Duplicates 🧐

# In[92]:


# Get total count of duplicated rows
df.duplicated().sum()


# In[93]:


# Find all duplicates
duplicates = df[df.duplicated(keep=False)]

# write duplicated rows
duplicates.to_csv('Duplciated rows.csv', index=False)


# In[94]:


# Remove duplicate rows
df = df.drop_duplicates()

# Check for duplicates
df.duplicated().sum()


# ## Look At Summary Data  📜

# In[95]:


# General stats for variables
df.describe()


# ## Check For Missing Values ⛔️

# In[96]:


# get percentage of null values
total_count = len(df)

# loop each column
for column in df.columns:
    null_count = df[column].isnull().sum()
    null_percentage = round((null_count / total_count) * 100,1)
    print(f"Column '{column}': {null_percentage} % of null values")


# ### Handle "Weapon Used Cd" ⛔️

# In[97]:


# Fill missing values (NaN) in 'Weapon Used Cd' column with 0
df.loc[df['Weapon Used Cd'].isna(), 'Weapon Used Cd'] = 0

# Fill missing values (NaN) in 'Weapon Desc' column with 'No weapons identified'
df.loc[df['Weapon Desc'].isna(), 'Weapon Desc'] = 'No weapons identified'


# In[98]:


# Filter the DataFrame 'df' to exclude rows where 'Vict Age' is either 0 or NaN
df = df[(df['Vict Age'] != 0) & (df['Vict Age'].notna())]
df.loc[:,'Vict Age'].head(10)


# ### Handle "Vict Sex" ⛔️

# In[99]:


# Filter the DataFrame 'df' to exclude rows where 'Vict Sex' is 'X' (Unknown), 'H' (invalid), or NaN
df = df[(df['Vict Sex'] != 'X') & (df['Vict Sex'] != 'H')&(df['Vict Sex'].notna())]

df.loc[: ,'Vict Sex'].head(10)


# In[100]:


print(df['Vict Descent'].unique())


# In[101]:


# Filter the DataFrame 'df' to exclude rows where 'Vict Descent' is '-' or missing (NaN)
df = df[(df['Vict Descent'] != '-') & (df['Vict Descent'].notna())]
df.loc[: ,'Vict Descent'].head(10)


# In[102]:


# Print dataframe info
print(df.info())


# In[103]:


# Drop rows with missing values (NaN) from the DataFrame 'df'
df = df.dropna()


# In[104]:


print(df.shape)


# ## Visualize Data 📊
# 

# In[105]:


# Check cross-correlation of numerical columns (use a spearman method because it is highly likely relationships are non-linear)
df_corr = df[cols_numerical].corr(method='spearman')
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df_corr,
            mask=np.triu(np.ones_like(df_corr, dtype=bool)), 
            cmap=sns.diverging_palette(230, 20, as_cmap=True), 
            vmin=-1.0, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[106]:


# Plot Histograms
df[cols_numerical].hist(xlabelsize =6)
plt.tight_layout()


# In[107]:


# Making Scattetrd Plots
df[cols_numerical].boxplot(rot=90)


# ## Identifying Ouliers 🔕

# In[108]:


# Create a box plot for Victim Age
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Vict Age'])
plt.xlabel("Victim Age")
plt.title("Box Plot of Victim Age with Outliers")
plt.show()

# Note that the plot below do not represesnt all ouliers on the variable 
# since it is possible for that old people could have been victims of a crime.


# ## Analyzing Variables 📈

# In[109]:


features_of_interest = ["Vict Age"]

# Making a nuberical summary
numerical_summary = df[features_of_interest].describe()
mode_age = df['Vict Age'].mode().values[0]
print(numerical_summary.loc[['mean', 'std']])
print(f"Mode of Victim Age: {mode_age}")


# In[110]:


# Create a histogram to visualize the spread of Victim Age
plt.figure(figsize=(8, 6))
sns.histplot(df['Vict Age'], bins=20, kde=True)
plt.xlabel("Vict Age")
plt.ylabel("Frequency")
plt.title("Histogram of Victim Age")
plt.show()


# In[111]:


# Calculate the z-score for Victim Age
z_scores = np.abs((df['Vict Age'] - df['Vict Age'].mean()) / df['Vict Age'].std())

# Define a threshold (e.g., 3 standard deviations)
threshold = 3

# Identify potential outliers
potential_outliers = df[z_scores > threshold]

# Calculate the percentage of outliers
percentage_outliers = (len(potential_outliers) / len(df)) * 100

print(f"Percentage of Outliers in Victim Age: {percentage_outliers:.2f}%")


# ### Handle "Vict Sex" ⛔️

# In[112]:


# Get the victim gender counts
victim_gender_counts = df['Vict Sex'].value_counts()

#create a pie chart of the victim gender counts
plt.figure(figsize=(6, 6))
plt.pie(victim_gender_counts, labels=victim_gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Victim Gender Distribution")
plt.show()


# In[113]:


temp = df['Vict Sex'].unique() # Get the unique values in the 'Vict Sex' column of the DataFrame 'df'
print(temp) # Print the unique values in the 'Vict Sex' column
print(len(df['AREA NAME'].unique())) # Print the number of unique values in the 'AREA NAME' column of the DataFrame 'df'


# ## Exploratory Data Analysis (EDA) 📈

# In[114]:


# Get the top 10 most frequent crime types and their counts from the 'Crm Cd' column
top_crime_types = df['Crm Cd'].value_counts().head(10)

# Create a new figure with a specified size (12 units wide and 6 units tall)
plt.figure(figsize=(12, 6))

# Create a horizontal bar plot using Seaborn, where 'x' represents counts and 'y' represents crime types
sns.barplot(x=top_crime_types.values, y=top_crime_types.index, orient="h")

plt.title("Top 10 Crime Types")# Set the title of the plot
plt.xlabel("Count") # Label the x-axis
plt.ylabel("Crime Type") # Label the y-axis
plt.show() # Display the plot


# In[115]:


# Filter data for male victims
male_victims = df[df['Vict Sex'] == 'M']

# Filter data for female victims
female_victims = df[df['Vict Sex'] == 'F']

# Count the occurrences of each crime code description for males and females
male_crime_counts = male_victims['Crm Cd'].value_counts()
female_crime_counts = female_victims['Crm Cd'].value_counts()

# Get the most common crime for each gender
most_common_male_crime = male_crime_counts.idxmax()
most_common_female_crime = female_crime_counts.idxmax()

# Create a bar chart to show the most common crimes by gender
plt.figure(figsize=(12, 6))
plt.bar(['Male', 'Female'], [male_crime_counts[most_common_male_crime], female_crime_counts[most_common_female_crime]], color=['blue', 'pink'])
plt.title("Most Common Crimes by Gender")
plt.xlabel("Gender")
plt.ylabel("Crime Count")
plt.text(0, male_crime_counts[most_common_male_crime] , most_common_male_crime, ha='center', va='bottom')
plt.text(1, female_crime_counts[most_common_female_crime] , most_common_female_crime, ha='center', va='bottom')
plt.show()


# In[116]:


# Get the counts of victims by descent/race from the 'Vict Descent' column
victim_gender_counts = df['Vict Descent'].value_counts()

# Create a new figure with a specified size (12 units wide and 6 units tall)
plt.figure(figsize=(12, 6))

# Get the counts of victims by race
crime_counts_by_race = df['Vict Descent'].value_counts()

# Generate a range of colors using the 'Paired' colormap for the pie chart
colors = plt.cm.Paired(range(len(crime_counts_by_race)))

plt.pie(crime_counts_by_race, startangle=140, colors=colors) # Create a pie chart with crime counts by race, starting the angle at 140 degrees
plt.title("Crime Distribution by Victim Race") # Set the title of the pie chart

# Create a legend with colors and percentages
legend_labels = [f"{race} ({count} - {percentage:.1f}%) " for race, count, percentage in zip(crime_counts_by_race.index, crime_counts_by_race.values, (crime_counts_by_race / crime_counts_by_race.sum()) * 100)]
plt.legend(legend_labels, loc="best", bbox_to_anchor=(1, 1))

plt.show() #Display the plot


# In[117]:


# Create a new figure with a specified size (12 units wide and 6 units tall)
plt.figure(figsize=(12, 6))

# Generate a histogram of victim ages with specified bins and styling
hist, bins, _ = plt.hist(df['Vict Age'], bins=np.arange(0, 101, 5), edgecolor='k', alpha=0.7)

# Set the title of the histogram
plt.title("Histogram of Victim Ages")

# Label the x-axis and y-axis
plt.xlabel("Age")
plt.ylabel("Count")

plt.grid(True) # Display the grid on the plot

# Add counts on top of each bar
for i in range(len(bins) - 1):
    plt.text(bins[i] + 2.5, hist[i] + 100, str(int(hist[i])), fontsize=10, ha='center', va='bottom')

plt.xticks(np.arange(0, 101, 5))  # Set X-axis ticks at 5-year intervals
plt.xlim(0, 100)  # Set X-axis limits
plt.show()


# In[118]:


# Get the top 10 areas with the highest crime counts from the 'AREA NAME' column
top_crime_types = df['AREA NAME'].value_counts().head(10)

# Create a new figure with a specified size (12 units wide and 6 units tall)
plt.figure(figsize=(12, 6))

# Create a horizontal bar plot using Seaborn, where 'x' represents counts and 'y' represents areas
sns.barplot(x=top_crime_types.values, y=top_crime_types.index, orient="h")

# Set the title and label the x-axis and y-axis
plt.title("Top 10 Areas Crime")
plt.xlabel("Count")
plt.ylabel("Area")

plt.show() # Display the plot


# In[119]:


central_data = df[df['AREA NAME'] == 'Central']

# Calculate the count of male and female victims in the Central area
male_count = (central_data['Vict Sex'] == 'M').sum()
female_count = (central_data['Vict Sex'] == 'F').sum()

# Create a bar plot
plt.figure(figsize=(8, 4))
sns.barplot(x=['Male', 'Female'], y=[male_count, female_count])
plt.title("Crime Distribution in Central Area (Male vs. Female)")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# In[120]:


df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

# Extract the year from the "Date Occurred" column and create a new column "Year"
df['Year'] = df['DATE OCC'].dt.year

# Count the number of crimes for each year
crime_counts_by_year = df['Year'].value_counts().sort_index()

# Create a line plot for crime counts over the years
plt.figure(figsize=(10, 6))
plt.plot(crime_counts_by_year.index, crime_counts_by_year.values, marker='o', linestyle='-')
plt.title("Crime Counts Over the Years")
plt.xlabel("Year")
plt.ylabel("Crime Count")
plt.grid(True)
plt.show()


# In[121]:


# Filter the dataset to include only records with victim ages between 25 and 40
filtered_data = df[(df['Vict Age'] >= 25) & (df['Vict Age'] <= 40)]

# Create a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='AREA NAME', y='Vict Age',showmeans =True, data=filtered_data)
plt.xticks(rotation=90)
plt.title("Box Plot of Victim Ages by Area (Ages 25 to 40)")
plt.xlabel("Area Name")
plt.ylabel("Victim Age")
plt.show()


# In[122]:


# Calculate the top 10 most frequent Crime Codes
top_10_crime_codes = df['Crm Cd'].value_counts().head(10).index.tolist()

# Filter the dataset for the top 10 Crime Codes
filtered_data = df[df['Crm Cd'].isin(top_10_crime_codes)]

# Create a violin plot
plt.figure(figsize=(14, 8))
ax = sns.violinplot(x='Crm Cd', y='Vict Age', data=filtered_data, inner="quart")
plt.xticks(rotation=90)
plt.title("Violin Plot of Victim Age by Top 10 Crime Codes")
plt.xlabel("Crime Code")
plt.ylabel("Victim Age")

# Create a legend for mean and median
legend_labels = []

for crime_code in top_10_crime_codes:
    mean_value = filtered_data[filtered_data['Crm Cd'] == crime_code]['Vict Age'].mean()
    median_value = filtered_data[filtered_data['Crm Cd'] == crime_code]['Vict Age'].median()
    print("Code" ,crime_code, "Mean",mean_value, "Median",median_value)


# Display the legend
# plt.legend(legend_labels, title="Statistics", loc="upper left", ncol=3)
# plt.legend(labels=legend_labels, title="Statistics", loc="upper left", ncol=3)
# plt.ylim(0, 150)
plt.tight_layout()
plt.show()


# In[123]:


# Create a box plot for victim ages by area
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Vict Descent', y='Vict Age', showmeans=True, palette='viridis')
plt.title("Box Plot of Victim Ages by Race")
plt.xlabel("Victim Descent")
plt.ylabel("Victim Age")
plt.xticks(rotation=90)
plt.show()


# In[124]:


# Filter data for female victims
female_victims = df[df['Vict Sex'] == 'F']

# Count the occurrences of each crime code for female victims
crime_counts_female = female_victims['Crm Cd'].value_counts()
percentages = [(count / len(female_victims)) * 100 for count in crime_counts_female]

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(crime_counts_female, labels=None, autopct='', startangle=140)
plt.title("Crime Codes for Female Victims")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Create a legend with crime codes
# plt.legend(crime_counts_female.index, title="Crime Codes", loc="center left", bbox_to_anchor=(1, 0.5), ncol=4)
legend_labels = [f"{code}: {percent:.2f}%" for code, percent in zip(crime_counts_female.index, percentages)]
plt.legend(legend_labels, title="Crime Codes", loc="center left", bbox_to_anchor=(1, 0.5), ncol=4)

# plt.legend(crime_counts_female.index, title="Crime Codes", loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()


# In[125]:


# Filter data for female victims
male_victims = df[df['Vict Sex'] == 'M']

# Count the occurrences of each crime code for female victims
crime_counts_male = male_victims['Crm Cd'].value_counts()
percentages = [(count / len(male_victims)) * 100 for count in crime_counts_male]

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(crime_counts_male, labels=None, autopct='', startangle=140)
plt.title("Crime Codes for Male Victims")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Create a legend with crime codes
# plt.legend(crime_counts_female.index, title="Crime Codes", loc="center left", bbox_to_anchor=(1, 0.5), ncol=4)
legend_labels = [f"{code}: {percent:.2f}%" for code, percent in zip(crime_counts_male.index, percentages)]
plt.legend(legend_labels, title="Crime Codes", loc="center left", bbox_to_anchor=(1, 0.5), ncol=4)

# plt.legend(crime_counts_female.index, title="Crime Codes", loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()


# In[126]:


df['TIME OCC'] = pd.to_datetime(df['TIME OCC'], format='%H:%M')

# Define time intervals
intervals = [(pd.Timestamp('00:01:00').time(), pd.Timestamp('06:00:00').time()),
             (pd.Timestamp('06:01:00').time(), pd.Timestamp('12:00:00').time()),
             (pd.Timestamp('12:01:00').time(), pd.Timestamp('18:00:00').time()),
             (pd.Timestamp('18:01:00').time(), pd.Timestamp('23:59:59').time())]

# Create labels for the intervals
labels = ['00:01-06:00', '06:01-12:00', '12:01-18:00', '18:01-24:00']

# Extract the time component (hours and minutes)
df['Time'] = df['TIME OCC'].dt.time

# Define a custom categorization function
def categorize_time(time):
    for i, interval in enumerate(intervals):
        if interval[0] <= time <= interval[1]:
            return labels[i]
    return None

# Apply the custom categorization function to create the 'Time Interval' column
df['Time Interval'] = df['Time'].apply(categorize_time)

# Count the number of occurrences in each interval
crime_counts = df['Time Interval'].value_counts().reindex(labels, fill_value=0)

# Display the results
print(crime_counts)


# In[127]:


# Create a new figure with a specified size (8 units wide and 6 units tall)
plt.figure(figsize=(8, 6))

# Create a bar plot with labels and crime counts, using a sky blue color
plt.bar(labels, crime_counts, color='skyblue')

# Label the x-axis as 'Time Intervals' and y-axis as 'Number of Crimes'
plt.xlabel('Time Intervals')
plt.ylabel('Number of Crimes')

# Set the title
plt.title('Crime Distribution by Time Interval')

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Show the plot
plt.show()


# In[128]:


#Check the shape of the data frame to determine the number of rows and columns
df.shape


# ## Data Cleaning Extension - 1: Handle Outliers ✅

# In[129]:


# Create Function To Clean Outliers Using IQR Method
def handle_outliers(df):
    for col in df.select_dtypes(include='number').columns:
            # Identify Quartiles
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
    
            # Identify Upper And Lower Limit
            lower_lim = Q1 - 1.5 * IQR
            upper_lim = Q3 + 1.5 * IQR
    
            # Drop All Values Outside Lower/Upper Limit
            df = df[(df[col] >= lower_lim) & (df[col] <= upper_lim)]

    return df

df = handle_outliers(df)


# In[130]:


df.shape


# ## Data Cleaning Extension - 2: Examine Data Balance ⚖️

# In[131]:


df['Status'].value_counts()


# In[132]:


df['Status'].value_counts().plot(kind = 'bar')
plt.title("Distribution Of Classes (Crime Status)")
plt.xlabel("Crime Status")
plt.ylabel("Count")
plt.show()


# * Even though the data is heavily skewed, there is no need to upsample/downsample as this imbalance is representative of real-world data. Most crimes are still currently under investigation (IC). 

# In[133]:


# Remove 'CC' Column Since It Accounts For Such A Small Percentage Of Data
df = df.drop(df[df['Status'] == 'CC'].index)
df['Status'].value_counts()


# In[134]:


df.head()


# ## Feature Engineering 🛠️

# In[135]:


# Days To Holiday Column
def get_us_holidays(year):
    # Fixed-date holidays
    holidays = [
        datetime(year, 1, 1),    # New Year's Day
        datetime(year, 7, 4),    # Independence Day
        datetime(year, 11, 11),  # Veterans Day
        datetime(year, 12, 25),  # Christmas Day
    ]
    
    # Floating holidays
    # Martin Luther King Jr. Day (3rd Monday of January)
    mlk = datetime(year, 1, 1) + timedelta(days=(14 - datetime(year, 1, 1).weekday()) % 7 + 14)
    holidays.append(mlk)
    
    # Presidents' Day (3rd Monday of February)
    presidents_day = datetime(year, 2, 1) + timedelta(days=(14 - datetime(year, 2, 1).weekday()) % 7 + 14)
    holidays.append(presidents_day)
    
    # Memorial Day (last Monday of May)
    memorial_day = datetime(year, 5, 31)
    while memorial_day.weekday() != 0:
        memorial_day -= timedelta(days=1)
    holidays.append(memorial_day)
    
    # Labor Day (1st Monday of September)
    labor_day = datetime(year, 9, 1)
    while labor_day.weekday() != 0:
        labor_day += timedelta(days=1)
    holidays.append(labor_day)
    
    # Columbus Day (2nd Monday of October)
    columbus_day = datetime(year, 10, 1) + timedelta(days=(7 - datetime(year, 10, 1).weekday()) % 7 + 7)
    holidays.append(columbus_day)
    
    # Thanksgiving Day (4th Thursday of November)
    thanksgiving = datetime(year, 11, 1)
    thursdays = 0
    while thursdays < 4:
        if thanksgiving.weekday() == 3:
            thursdays += 1
        thanksgiving += timedelta(days=1)
    holidays.append(thanksgiving - timedelta(days=1))  # because it overshoots
    
    return holidays

# Ensure DATE OCC is datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

# Calculate Days_To_Holiday
def days_to_nearest_holiday(date):
    year = date.year
    holidays = get_us_holidays(year)
    return min(abs((date - h).days) for h in holidays)

df['Days_To_Holiday'] = df['DATE OCC'].apply(days_to_nearest_holiday)

plt.hist(df['Days_To_Holiday'])
plt.title("Count Of Days To Holiday")
plt.xlabel("Days To Holiday")
plt.ylabel("Count")


# In[136]:


# Seperate Date, Time, Month into Individual Columns
# Reported Date
df['RPTD_Year'] = df['Date Rptd'].dt.year
df['RPTD_Month'] = df['Date Rptd'].dt.month
df['RPTD_Day'] = df['Date Rptd'].dt.day

# Date Occured
df['OCC_Year'] = df['DATE OCC'].dt.year
df['OCC_Month'] = df['DATE OCC'].dt.month
df['OCC_Date'] = df['DATE OCC'].dt.day

# Time Occured
df['OCC_Hour'] = df['TIME OCC'].dt.hour
df['OCC_Minute'] = df['TIME OCC'].dt.minute
df['OCC_Second'] = df['TIME OCC'].dt.second

# Time Of Day Occured
def map_time_of_day(hour):
    if 0 <= hour < 5:
        return 'Late Night'
    elif 5 <= hour < 8:
        return 'Early Morning'
    elif 8 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['OCC_TimeOfDay'] = df['OCC_Hour'].apply(map_time_of_day)

# Drop Parsed Columns
df.drop(['Date Rptd', 'DATE OCC', 'TIME OCC', 'Time', 'Year'], axis = 1, inplace = True)


# ## Rename Columns 📝

# In[137]:


df.rename(columns={
    'AREA': 'area_code',
    'AREA NAME': 'area_name',
    'Rpt Dist No': 'reporting_district',
    'Part 1-2': 'crime_part',
    'Crm Cd': 'crime_code',
    'Crm Cd Desc': 'crime_description',
    'Mocodes': 'mo_codes',
    'Vict Age': 'victim_age',
    'Vict Sex': 'victim_sex',
    'Vict Descent': 'victim_descent',
    'Premis Cd': 'premise_code',
    'Premis Desc': 'premise_description',
    'Weapon Used Cd': 'weapon_code',
    'Weapon Desc': 'weapon_description',
    'Status Desc': 'status_description',
    'Target': 'arrest_type',
    'Time Interval': 'occ_time_interval',
    'Days_To_Holiday': 'days_to_holiday',
    'RPTD_Year': 'report_year',
    'RPTD_Month': 'report_month',
    'RPTD_Day': 'report_day',
    'OCC_Year': 'occurrence_year',
    'OCC_Month': 'occurrence_month',
    'OCC_Date': 'occurrence_day',
    'OCC_Hour': 'occurrence_hour',
    'OCC_Minute': 'occurrence_minute',
    'OCC_Second': 'occurrence_second',
    'OCC_TimeOfDay': 'occurrence_time_of_day',
}, inplace=True)


# In[138]:


df.info()


# ## Mo_Codes Handling 🚨
# * We can see below that `mo_codes` has a large number of unique values. This will cause pd.get_dummies() (One-Hot Encoding) fail. There are a few things we can do here:
#     * Create the count of `mo_codes`
#     * Top ***k*** binary flags (whether common flags are present or not (top ***k*** flags)
#     * Behavioral Clustering
#     * Sequence Embedding

# In[139]:


# Count The Number Mo_Codes
df['mo_code_count'] = df['mo_codes'].apply(lambda x: len(x.split()))

# Top K Binary Flags
from collections import Counter

def binary_flags(df, k=20):
    # Step 1: Flatten all codes into one list
    all_mo_lists = df['mo_codes'].dropna().apply(lambda x: str(x).split())
    flat_list = [code for sublist in all_mo_lists for code in sublist]
    print(flat_list)

    # Step 2: Count frequency of each code
    mo_counts = Counter(flat_list)

    # Step 3: Get top K codes
    top_k_codes = [code for code, _ in mo_counts.most_common(k)]

    # Step 4: Create binary flags
    for code in top_k_codes:
        df[f'mo_{code}'] = df['mo_codes'].apply(lambda x: int(code in str(x).split()))

    return df

df = binary_flags(df, k=20)


# In[140]:


# Cluster Codes
def cluster(k = 5):
    vectorizer = CountVectorizer(analyzer = str.split) # Tokenizes on Each Mo_Code Sep by ' '
    x_mo = vectorizer.fit_transform(df['mo_codes']) # Learn the Vocabulary 

    kmeans = KMeans(n_clusters = k, random_state = 42) # Assign Codes To Cluster
    df['mo_cluster'] = kmeans.fit_predict(x_mo) # Assign Clusters To Row on df

    pca = PCA(n_components = 2) # Dimensionality Reduction for visualization
    x_pca = pca.fit_transform(x_mo.toarray())

    # Scatter Plot Of Clusters
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=df['mo_cluster'], cmap='tab10', alpha=0.6)
    plt.title(f"KMeans Clustering of MO Codes (k={k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

cluster(10)


# In[142]:


df.drop(['status_description', 'Status'], inplace = True, axis = 1)


# ## Encode Columns 🧮

# In[143]:


for col in df.select_dtypes(include=['object', 'string']):
    print(f"{col}: {df[col].nunique()} unique values")


# In[144]:


df.columns


# In[145]:


def encode_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    for col in ['mo_codes', 'arrest_type']: # Ignore These Two Cols For Further Processing
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    df = pd.get_dummies(df, columns=categorical_cols) # One-Hot Encode Features

    # Label Encode Target
    le = LabelEncoder()
    df['arrest_type'] = le.fit_transform(df['arrest_type'])
    
    return df
    
df = encode_columns(df)


# ## Reassign Target To Last Column 🔁

# In[146]:


# Move Target Column To Last Position
df = df.assign(arrest_type=df.pop('arrest_type'))


# In[147]:


# View Cleaned DF
df.head()


# In[148]:


# Save CSV
df.to_csv("../Data/Cleaned_Dataset.csv")

