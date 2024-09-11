import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#dataset has songs only from 1922 - 2021

#set display options to show all columns, if i dont do this it shows ... instead of everything
pd.set_option('display.max_columns', None)

df_tracks = pd.read_csv("tracks.csv")

#check first 5 rows
print(df_tracks.head())

#check total num of null values in dataset
print(pd.isnull(df_tracks).sum())

#total number of rows + columns in dataset, check datatype and memory usage
print(df_tracks.info())

#which artist has the least number of followers in spotify?
#top 10 least popular songs
sorted_df = df_tracks.sort_values("popularity", ascending = True).head(10)
print(sorted_df)

#descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
print(df_tracks.describe().transpose())

#top 10 most popular songs which are >90
most_popular = df_tracks.query("popularity>90", inplace = False).sort_values("popularity", ascending = False)
print(most_popular[:10])

#next cell
df_tracks.set_index("release_date", inplace = True)
#handling errors and mixed formats
df_tracks.index = pd.to_datetime(df_tracks.index, format="%Y-%m-%d", errors='coerce')
#check any Not a Time (NaT) vals in the index
print(df_tracks[df_tracks.index.isna()])
print(df_tracks.head())

#artist in 18th row = victor boucher
print(df_tracks[["artists"]].iloc[18])

#convert duration in miliseconds to seconds
df_tracks["duration"] = df_tracks["duration_ms"].apply(lambda x: round(x/1000))
df_tracks.drop("duration_ms", inplace = True, axis = 1)
print(df_tracks.duration.head())

#next cell, creating the first visualization that is a correlation map
#drop non numeric columns columns and apply pearson correlation method
df_tracks_numeric = df_tracks.drop(["key", "mode", "explicit"], axis=1)
df_tracks_numeric = df_tracks_numeric.select_dtypes(include=[np.number])
corr_df = df_tracks_numeric.corr(method="pearson")

#customize correlation map
plt.figure(figsize=(14, 6))
heatmap = sns.heatmap(corr_df, annot=True, fmt=".1g", vmin=-1, vmax=1, center=0, cmap="inferno", linewidths=0.5)
heatmap.set_title("Correlation HeatMap Between Variables")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
plt.show()

#regression plots
#0.4% of my total dataset is 2346 rows
sample_df = df_tracks.sample(int(0.004*len(df_tracks)))
print(len(sample_df))

#regression plot between loudness and energy
plt.figure(figsize = (10,6))
sns.regplot(data = sample_df, y = "loudness", x = "energy", color = "c").set(title = "Loudness vs Energy Correlation")
plt.show()

#regression plot between popularity and acousticness
plt.figure(figsize = (10,6))
sns.regplot(data = sample_df, y = "popularity", x = "acousticness", color = "b").set(title = "Popularity vs Acousticness Correlation")
plt.show()

df_tracks = pd.read_csv("tracks.csv")

#function to fix ambiguous dates
def fix_dates(date_str):
    if pd.isna(date_str):
        return np.nan
    #try parsing different date formats
    for fmt in ('%Y-%m-%d', '%m/%d/%y', '%Y'):
        try:
            date = pd.to_datetime(date_str, format=fmt, errors='coerce')
            if date is not pd.NaT:
                #if date is valid but out of the expected range, adjust the year
                if date.year > 2021:
                    date = date.replace(year=date.year - 100)
                return date
        except ValueError:
            continue
    return np.nan

#apply the date correction, drop rows with missing or invalid dates
df_tracks['release_date'] = df_tracks['release_date'].apply(fix_dates)

df_tracks = df_tracks.dropna(subset=['release_date'])
df_tracks = df_tracks[df_tracks['release_date'].dt.year.between(1922, 2021)]

#set release_date as the index
df_tracks.set_index('release_date', inplace=True)

#reset the index because we need 'release_date' as a column again
df_tracks.reset_index(inplace=True)
df_tracks.to_csv("tracks_cleaned.csv", index=False)

print("Cleaned CSV file has been saved as 'tracks_cleaned.csv'.")