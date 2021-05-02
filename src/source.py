titles = ("Modern Family", "The Big Bang Theory", "Friends", "The Good Place", "Brooklyn Nine-Nine")

## Libraries
# The standard libraries for advanced data frame and numerical processing.
import pandas as pd
import numpy as np

# The libaries that I'll be using for my plots
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# I used the scipy package for a smoothing function that I'll explain later in this document
import scipy as sp

# And finally, the IMDb package used to extract the ratings data
from imdb import IMDb


## Extracting the IMDb ratings
### Building the functions
## Sets up the basic use of the IMDb library, if one hasn't already been created.
def imdb_setup():
  from imdb import IMDb
  return IMDb()
 
## Function to search the IMDB database for TV Series and return the movieID number.
 #   title = the TV series to search
 #   ia = the imdb link. If none provided, this function will create one.
def imdb_search(title, ia = None):

  # If no IA provided, create one using the function above
  if ia == None:
    ia = imdb_setup()

  # Search the IMDB database, using the IMDb library and obtain the first result, most popular result
  # This can easily be used to search the list and pick one, however for automation, it'll pick the most likely case
  search_imdb = ia.search_movie(title = title)
  
  # Use the search results to obtain the movieID reference, to be used in other functions.
  movie_ID    = search_imdb[0].movieID
  return movie_ID

## Function to extract the IMDb series object. This contains the ratings, among other information.
 # movie_ID = the movie ID unique identifer used by IMDB for this particular TV Series
 # episodes = Boolean. Return the ratings for each episode? This takes a bit longer to extract.
def imdb_series(movie_ID, episodes = True, ia = None):
  if ia == None:
    ia = imdb_setup()
  if episodes:
    series = ia.get_movie(movie_ID, info = ("episodes"))
  else:
    series = ia.get_movie(movie_ID)
  return series

## A function that allows us to obtain the details of each episodes
 # series = IMDb movie object, returned in imdb_series
 # demongraphics = Boolean, to return ratings by gender
def imdb_episodes(series, demographics = False, ia = None):
  # If no IA provided, create one using the function above
  if ia == None:
    ia = imdb_setup()
  
  # create a dataframe with the columns we want to return
  df = pd.DataFrame(columns = ["season", "episode", "aired", "title", "rating"])
  
  # loop through every episode of every season to obtain required information
  for seasons in series.data["episodes"]:
    season = series.data["episodes"][seasons]
    for episodes in season:
      episode = series.data["episodes"][seasons][episodes]
      
      sea           = episode.get("season")
      epi           = episode.get("episode")
      air           = pd.to_datetime(episode.get("original air date"))
      title         = episode.get("title")
      rating        = episode.get("rating")
    
      new_row = {"title" : title, "rating" : rating, "season" : sea, "episode" : epi, "aired" : air}
    
      # If demogragics = True, obtain the individual male/female ratings for each episode
      if demographics:
        epi_vote_det  = ia.get_movie_vote_details(episode.movieID)
        rating_male   = epi_vote_det["data"]["demographics"]["males"]["rating"]
        rating_female = epi_vote_det["data"]["demographics"]["females"]["rating"]
        new_row["rating_male"] = rating_male
        new_row["rating_female"] = rating_female

      df = df.append(new_row, ignore_index = True)
      df["row"] = np.arange(df.shape[0])
    
  return df


# Function that will take all of the IMDB functions and extract/return a final dataset.
# This can be used for a single TV series, or a tuple of many.
def extract_TV_Ratings(titles = "", demographics = False):
  # Create an empty dataframe, which will be used to store the data
  df = pd.DataFrame()
  
  # Identify if the type of data is a string or tuple. 
  #  If string, search for the single TV series. 
  #  If tuple, loop through each TV series.
  if type(titles) == str:
    movie_ID     = imdb_search(titles)
    series       = imdb_series(movie_ID)
    df           = imdb_episodes(series, demographics)
    df["series"] = titles

  if type(titles) == tuple:
    for title in titles:
      movie_ID       = imdb_search(title)
      series         = imdb_series(movie_ID)
      temp           = imdb_episodes(series, demographics)
      temp["series"] = title
      df = df.append(temp, ignore_index = True)
  return df

# Use the extract_TV_Ratings function to extract the TV series data of the titles object.
#   extracting many TV series will take some time
data = extract_TV_Ratings(titles, demographics=False)
data.head()

## What does the data tell us?

#### Mean rating per season
# Create a new dataframe, containing the mean for each season, removing an episodes that don't have a rating (you often see NA's in future or unaired episodes)
season_mean = data
season_mean = season_mean.groupby(["series", "season"], as_index = False)["rating"].mean()
season_mean = season_mean[season_mean["rating"].notna()]

# Create a smooth plot based on the mean ratings per season
# Start by creating an empty dataframe, containing just the data needed for the plot.
chart_data = pd.DataFrame(columns = ["x", "y", "series"])
# Then loop through each TV Series
for series in season_mean["series"].unique():
  df = season_mean[season_mean["series"] == series]
  x = df["season"]
  y = df["rating"]
  
  # Use the make_interp_spline function to smooth the plot by having 300 points on the x axis.
  a_BSpline = sp.interpolate.make_interp_spline(x, y)
  new_x = np.linspace(1, len(df), 300)
  new_y = a_BSpline(new_x)
  dt    = {"x" : new_x, "y" : new_y }
  dt    = pd.DataFrame(dt)
  dt["series"] = series
  chart_data = chart_data.append(dt, ignore_index = True)

# Plot the smoothed data
_ = sns.lineplot(x = "x", y = "y", hue = "series", data = chart_data, linewidth = 2, alpha = 0.8).set(ylim=(7,10), xlabel = "Season", ylabel = "Rating", title = "Mean Ratings per Season for each TV Series")
plt.show()
plt.clf()

# Distribution of ratings
p = sns.kdeplot(x = "rating", data = data, hue = "series", alpha = 0.4)
_ = p.set(xlim=(5, 10), xlabel = "Rating", title = "Distribution of Ratings per TV Series")
plt.show()
plt.clf()

# Violin plot of all episodes
violin = sns.violinplot(x = "series", y = "rating", data = data, alpha = 0.8)
violin = violin.set(ylim=(5,10), title = "Distribution of ratings", xlabel = "TV Series", ylabel = "Rating")
plt.show()
plt.clf()

# Barcode
def norm_fill(df, size = None):
  if size == None: no_row = len(df)
  else: no_row = size
  no_col = len(df.columns)
  
  new_df = pd.DataFrame(index = range(no_row), columns=df.columns)

  for col in df:
    colSeriesObj = df[col]
    mx = no_row
    le = len(colSeriesObj.dropna())

    diff = mx/le
    
    for index in range(le):
      old = int(index)
      new = int(old*diff)
      new_df.loc[new, col] = df.loc[old,col]
    
  return new_df.fillna(method = "ffill")

heatmap = data[data["season"] <= 99][["series", "row", "rating"]].pivot(index = "row", columns = "series", values = "rating")

def plot_func(df, size = None, square = False, linewidth = 0.0):
  # Use the previously defined function to create a new normalised data frame
  df = norm_fill(df, size)
  
  # I then use the matplotlib TwoSlopeNorm function to create a diverging color palette. This allows me to have a two colored gradiant from medium to highly rated.
  # The lowest value is 6 (any ratings below 6 will have the same color) and the maximum value is 10, as no rating can exceed this. I then set the centre (where the colors start to change) as 7.5.
  divnorm = TwoSlopeNorm(vmin=6, vcenter=7.5, vmax=10)
  cmap = sns.diverging_palette(50, 200, 50, 50, as_cmap=True)

  # I then create the plot, with a figure size of 16 by 8 and add the titles and labels.
  fig,ax = plt.subplots(1,1,figsize=(16,8))
  _ = sns.heatmap(df.T, cmap = cmap, linewidth = linewidth, cbar_kws={"shrink" : 0.6}, square = square, norm = divnorm, ax = ax)
  _ = _.set(ylabel = None, xlabel = "Episodes", xticklabels = [], title="Ratings per episode")
  
  # I then adjust the plot sightly as this seems to fit better
  ax.figure.subplots_adjust(left = 0.2)
  
  # Finally, I amend the color labels. Rather than show ratings 6 to 10, I display the words "medium rated" and "highly rated" in the middle of the colors.
  colorbar = ax.collections[0].colorbar
  colorbar.set_ticks([6.75,9])
  colorbar.set_ticklabels(['medium rated','highly rated'])
  plt.show()
  plt.clf()

plot_func(heatmap)

## The final plot
heatmap = data[data["season"] <= 99][["series", "row", "rating"]].pivot(index = "row", columns = "series", values = "rating")

plot_func(heatmap, size = 40, square = True, linewidth = 0.5)


###### Bonus Plot
bonus_titles = ("Game of Thrones")
bonus_data = extract_TV_Ratings(bonus_titles, demographics=False)

bonus_heatmap = bonus_data[bonus_data["season"] <= 99][["series", "row", "rating"]].pivot(index = "row", columns = "series", values = "rating")

plot_func(bonus_heatmap, square = False, linewidth = 0.2)
