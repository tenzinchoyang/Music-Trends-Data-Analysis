import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
df_genre = pd.read_csv("SpotifyFeatures.csv")
print(df_genre.head())

#duration of songs for different genres

plt.title("Duration of the Songs in Different Genres")
sns.color_palette("rocket", as_cmap = True)
sns.barplot(y = "genre", x = "duration_ms", data = df_genre)
plt.xlabel("Duration in milli seconds")
plt.ylabel("Genres")
plt.show()

#top 5 genres by popularity
sns.set_style(style = "darkgrid")
plt.figure(figsize = (10,5))
famous = df_genre.sort_values("popularity", ascending = False).head(10)
sns.barplot(y="genre", x="popularity", data = famous).set(title = "Top 5 Genres by Popularity")
plt.show()