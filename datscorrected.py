import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import warnings
import os


# Define file containing dataset
file_path = os.path.join(os.path.dirname(__file__),'/Users/vandana/Desktop/Analyze Your Runkeeper Fitness Data/datasets/cardioActivities.csv')


# Load the CSV file with parse_dates and index_col parameters 
df_activities = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Print the DataFrame and column names
print("Initial DataFrame:")
print(df_activities.head())
print("Column names:", df_activities.columns)

# If 'Date' is the index, drop rows with NaT values in the index
df_activities = df_activities.loc[~df_activities.index.isna()]

# Rename columns if necessary
df_activities.rename(columns=lambda x: x.strip(), inplace=True)

# Print column names again to verify if renaming was successful
print("Updated column names:", df_activities.columns)

# Print the first few rows to verify
print("DataFrame after dropping rows with invalid dates:")
print(df_activities.head())

# Ensure the 'Date' index is in datetime format (it should already be if parsed correctly)
df_activities.index = pd.to_datetime(df_activities.index, errors='coerce')

# Drop rows with invalid dates after re-confirming index as datetime
df_run = df_activities.loc[~df_activities.index.isna()]
print("Invalid dates dropped")

# Ensure index is sorted
df_run=df_run.sort_index()
print("Index is sorted")

# Print the first few rows to verify
print("First few rows of the sorted DataFrame:")
print(df_run.head())

# Check the date range of the DataFrame
print("Date Range in DataFrame:")
print(df_activities.index.min(), df_activities.index.max())

# Print initial DataFrame information
print("Initial DataFrame:")
print(df_activities.head())

# Check the date range of the DataFrame
print("Date Range in DataFrame:")
print(df_activities.index.min(), df_activities.index.max())

# Print 3 random rows
print("Sample Rows:")
print(df_activities.sample(3))

# Print DataFrame information
print("DataFrame Info:")
print(df_activities.info())

# Drop unnecessary columns
cols_to_drop = ['Friend\'s Tagged', 'Route Name', 'GPX File', 'Activity Id', 'Calories Burned', 'Notes']
df_activities = df_activities.drop(columns=cols_to_drop)
print("unnecessary columns dropped")

# Count types of training activities
activity_count = df_activities['Type'].value_counts()
print("Activity Count:")
print(activity_count)

# Rename 'Other' to 'Unicycling'
df_activities['Type'] = df_activities['Type'].replace('Other', 'Unicycling')
print("Unique Activity Types After Renaming:")
print(df_activities['Type'].unique())

# Count missing values
missing_value_count = df_activities.isna().sum()
print("Missing Values Count:")
print(missing_value_count)

# Calculate average heart rate for each training activity type
avg_hr = df_activities.groupby('Type')['Average Heart Rate (bpm)'].mean()
print("Average Heart Rates:")
print(avg_hr)

# Split DataFrame by activity type
df_run = df_activities[df_activities['Type'] == 'Running'].copy()
df_cycle = df_activities[df_activities['Type'] == 'Cycling'].copy()
df_walking = df_activities[df_activities['Type'] == 'Walking'].copy()
df_unicycling = df_activities[df_activities['Type'] == 'Unicycling'].copy()

# Fill missing values with calculated means
df_run['Average Heart Rate (bpm)'] = df_run['Average Heart Rate (bpm)'].fillna(avg_hr['Running'])
df_walking['Average Heart Rate (bpm)'] = df_walking['Average Heart Rate (bpm)'].fillna(110)
df_cycle['Average Heart Rate (bpm)'] = df_cycle['Average Heart Rate (bpm)'].fillna(avg_hr['Cycling'])
df_unicycling['Average Heart Rate (bpm)'] = df_unicycling['Average Heart Rate (bpm)'].fillna(avg_hr['Unicycling'])

# Check for missing values after filling
print("Missing Values Count for Each DataFrame:")
print("Running DataFrame:", df_run.isna().sum())
print("Walking DataFrame:", df_walking.isna().sum())
print("Cycling DataFrame:", df_cycle.isna().sum())
print("Unicycling DataFrame:", df_unicycling.isna().sum())

# Plotting style
plt.style.use('ggplot')
warnings.filterwarnings(action='ignore', module='matplotlib.figure', category=UserWarning, message='This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.')

# Subset data for the period 2013 to 2018
start_date = '2013-01-01'
end_date = '2018-12-31'
mask = (df_run.index >= start_date) & (df_run.index <= end_date)
runs_subset_2013_2018 = df_run.loc[mask]

print("Running DataFrame Subset (2013-2018):")
print(runs_subset_2013_2018.head())

# Plot Running Distance Over Time (2013-2018)
plt.figure(figsize=(12, 6))
plt.plot(runs_subset_2013_2018.index, runs_subset_2013_2018['Distance (km)'], 'o', markersize=3)
plt.title('Running Distance Over Time (2013-2018)')
plt.xlabel('Date')
plt.ylabel('Distance (km)')
plt.grid(True)
plt.show()

# Subset data for the period 2015 to 2018
start_date = '2015-01-01'
end_date = '2018-12-31'
mask = (df_run.index >= start_date) & (df_run.index <= end_date)
runs_subset_2015_2018 = df_run.loc[mask]

print("Running DataFrame Subset (2015-2018):")
print(runs_subset_2015_2018.head())

# Convert 'Duration' from H:MM:SS to total seconds
def duration_to_seconds(duration):
    try:
        parts = list(map(int, duration.split(':')))
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None
    
# Make a copy of the subset DataFrame 
runs_subset_2015_2018 = runs_subset_2015_2018.copy()

# Add a new column for duration in seconds
runs_subset_2015_2018.loc[:,'Duration_seconds'] = runs_subset_2015_2018['Duration'].apply(duration_to_seconds)

# Apply conversion to 'Duration_seconds'
runs_subset_2015_2018 = runs_subset_2015_2018.copy()
runs_subset_2015_2018['Duration_seconds'] = runs_subset_2015_2018['Duration'].apply(duration_to_seconds)

print("Duration converted to seconds.")


# Plot Running Distance Over Time (2015-2018)
plt.figure(figsize=(12, 6))
plt.plot(runs_subset_2015_2018.index, runs_subset_2015_2018['Distance (km)'], 'o', markersize=3)
plt.title('Running Distance Over Time (2015-2018)')
plt.xlabel('Date')
plt.ylabel('Distance (km)')
plt.grid(True)
plt.show()
print("Plot for running distance over time (2015-2018) created.")

# Convert index to DatetimeIndex with error handling
def convert_to_datetime_index(df):
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')  # Convert with error handling
    except Exception as e:
        print(f"Error converting index to datetime: {e}")
    return df


#Apply conversion
runs_subset_2015_2018 = convert_to_datetime_index(runs_subset_2015_2018)

# Drop rows where index is NaT
runs_subset_2015_2018 = runs_subset_2015_2018.loc[~runs_subset_2015_2018.index.isna()]

# Ensure columns are numeric
def convert_columns_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, setting errors to NaN
    return df

# List of columns that should be numeric
numeric_columns = [
    'Distance (km)',
    'Duration_seconds',
    'Average Pace',
    'Average Speed (km/h)',
    'Climb (m)',
    'Average Heart Rate (bpm)'
]

# Convert columns to numeric
runs_subset_2015_2018 = convert_columns_to_numeric(runs_subset_2015_2018, numeric_columns)

# Extract the year from the DatetimeIndex
runs_subset_2015_2018['Year'] = runs_subset_2015_2018.index.year

# Calculate annual statistics
annual_stats = runs_subset_2015_2018.groupby('Year').agg({
    'Distance (km)': ['mean', 'sum', 'count'],
    'Duration_seconds': 'mean'
}).reset_index()
print('Annual Statistics (2015-2018):')
print(annual_stats)

# Calculate weekly statistics
weekly_stats = runs_subset_2015_2018.resample('W').agg({
    'Distance (km)': ['mean', 'sum'],
    'Duration_seconds': ['mean', 'sum'],
    'Average Pace': 'mean',
    'Average Speed (km/h)': 'mean',
    'Climb (m)': 'mean',
    'Average Heart Rate (bpm)': 'mean'
})
print('Weekly Statistics (2015-2018):')
print(weekly_stats.head())

# Calculate weekly counts
weekly_counts = runs_subset_2015_2018.resample('W').size()

# Calculate mean weekly counts
weekly_counts_average = weekly_counts.mean()
print('Average Trainings Per Week:', weekly_counts_average)


# Prepare data for plotting
runs_distance = runs_subset_2015_2018['Distance (km)']
runs_hr = runs_subset_2015_2018['Average Heart Rate (bpm)']

# Check if data series are not empty
if runs_distance.empty or runs_hr.empty:
    print("One or both of the series to plot are empty.")
else:
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot and customize the first subplot (Distance)
    ax1.plot(runs_distance.index, runs_distance, color='blue', marker='o', linestyle='-', markersize=4, label='Distance (km)')
    ax1.axhline(runs_distance.mean(), color='blue', linestyle='-.', linewidth=1, label='Average Distance')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Historical data with averages')
    ax1.legend()
    ax1.grid(True)

    # Plot and customize the second subplot (Heart Rate)
    ax2.plot(runs_hr.index, runs_hr, color='green', marker='x', linestyle='-.', markersize=4, label='Heart Rate (bpm)')
    ax2.axhline(runs_hr.mean(), color='grey', linestyle='--', linewidth=1, label='Average Heart Rate')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Average Heart Rate (bpm)')
    ax2.legend()
    ax2.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show plot with tight layout
    plt.tight_layout()
    plt.show()
    print("plot for 'Historical data with averages' created")


#****   TASK 7  ********
# Prepare data
df_run_dist_annual = df_run.resample('YE').agg({'Distance (km)': 'sum'})
df_run_dist_annual['Year'] = df_run_dist_annual.index.year

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot and customize
df_run_dist_annual.plot(x='Year', y='Distance (km)', marker='*', markersize=14, linewidth=0, color='blue', ax=ax)

# Set limits and labels
ax.set_ylim([0, 1210])
ax.set_xlim(['2012', '2019'])
ax.set_ylabel('Distance (km)')
ax.set_xlabel('Years')
ax.set_title('Annual Totals for Distance')

# Add shaded regions for targets
ax.axhspan(1000, 1210, color='green', alpha=0.4, label='Goal Met')
ax.axhspan(800, 1000, color='yellow', alpha=0.3, label='Near Goal')
ax.axhline(y=1000, color='black', linestyle='--', label='1000 km Goal')

# Add legend
ax.legend()

plt.show()

if 'Distance (km)' not in df_activities.columns:
    raise ValueError("The DataFrame must contain a 'distance' column")

# Resample data to weekly frequency if not already
df_run_dist_wkly = df_activities['Distance (km)'].resample('W').sum()  # or use mean(), etc., depending on your analysis needs

# Decompose the time series
decomposed = sm.tsa.seasonal_decompose(df_run_dist_wkly, model='additive', period=52)

# Create plot
fig = plt.figure(figsize=(10, 8))

# Plot and customize
ax = decomposed.trend.plot(label='Trend', linewidth=2)
decomposed.observed.plot(label='Observed', linewidth=0.5, ax=ax)

ax.legend()
ax.set_title('Running Distance Trend')

# Show plot
plt.show()
print('plot for Running Distance Trend created')





df_run_hr_all = df_activities['Average Heart Rate (bpm)']





# Define heart rate zones and their corresponding names and colors
hr_zones = [0, 100, 125, 133, 142, 151, 173, 200]  # 8 boundaries define 7 bins (zones)
zone_names = ['Very Easy', 'Easy', 'Moderate', 'Hard', 'Very Hard', 'Maximal', 'Maximal Plus']  # 7 zone names
zone_colors = ['blue', 'green', 'yellow', 'orange', 'tomato', 'red', 'purple']

# Check if the number of zones matches the number of colors
if len(hr_zones) - 1 != len(zone_colors):
    raise ValueError("Number of hr_zones should be one more than the number of zone_colors")

# Create the plot
fig, ax = plt.subplots()

# Plot histogram
n, bins, patches = ax.hist(df_run_hr_all, bins=hr_zones, alpha=0.5, edgecolor='black')

# Print the number of patches and zone colors to debug
print(f'Number of patches: {len(patches)}')
print(f'Number of zone colors: {len(zone_colors)}')

# Color the patches
for i in range(len(patches)):
    patches[i].set_facecolor(zone_colors[i])

# Customize the plot
ax.set_title('Distribution of Heart Rate')
ax.set_ylabel('Number of Runs')
ax.set_xlabel('Heart Rate (bpm)')

# Calculate midpoints for the tick labels
midpoints = [(hr_zones[i] + hr_zones[i+1]) / 2 for i in range(len(hr_zones)-1)]

# Set tick positions and labels correctly
ax.set_xticks(midpoints)  # Set ticks at midpoints of each zone
ax.set_xticklabels(zone_names, rotation=45, ha='right')

# Display the plot
plt.show()


# Concatenate the DataFrames
df_run_walk_cycle = pd.concat([df_run, df_walking, df_cycle], ignore_index=True)

# Define columns for distance, climb, and speed
dist_climb_cols = ['Distance (km)', 'Climb (m)']
speed_col = ['Average Speed (km/h)']

# Calculate total distance and climb for each type of activity
df_totals = df_run_walk_cycle.groupby('Type')[dist_climb_cols].sum().reset_index()

print('Totals for different training types:')
print(df_totals)

# Calculate summary statistics for each type of activity
df_summary = df_run_walk_cycle.groupby('Type')[dist_climb_cols + speed_col].describe()

# Add totals to the summary statistics
for col in dist_climb_cols:
    df_summary[(col, 'total')] = df_totals.set_index('Type')[col]

print('Summary statistics for different training types:')
print(df_summary)

# Optionally, save the summary to a CSV file
df_summary.to_csv('summary_statistics.csv')




#FunFacts
# Given data
my_total_km = 5224
my_shoes = 7
forrest_total_km = 24700

# Calculate average kilometers per pair of shoes
average_shoes_lifetime = my_total_km / my_shoes

# Calculate the number of shoes needed for Forrest Gump's total distance
shoes_for_forrest_run = forrest_total_km / average_shoes_lifetime

# Print the result
print('Forrest Gump would need {:.2f} pairs of shoes!'.format(shoes_for_forrest_run))
