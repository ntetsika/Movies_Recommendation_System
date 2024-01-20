import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
import mercury as mr
warnings.simplefilter(action='ignore', category=FutureWarning)

def addlabels(x,y):
    '''Adds label in the bars plot
       
    Parameter:  x: list: x-coordinates
                y: list: y-coordinates
    Return:       None
    '''
    for i in range(len(x)):
        plt.text(y[0]*0.5, i, x[i], ha = 'center' )

def plot_bars(x,y, x_label, title):
    '''Prints bars for the most rated movies
       
    Parameter:  x: list: movie titles
                y: list: ratings
                x_label: str: name of x-axis label
                title: str: plot's title
    Return:       None
    '''
    fig, ax = plt.subplots()

    colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
    # Save the chart so we can loop through the bars below.
    bars = ax.barh(
        y=x,
        width=y,
        color=colors
    )

    addlabels(x,y)

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False, labelleft = False, labelbottom = False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Add text annotations to the top of the bars.
    bar_color = bars[0].get_facecolor()
    for bar in bars:
      ax.text(
          bar.get_x() + bar.get_width()*1.05,
          bar.get_y() + 0.3,
          round(bar.get_width(), 2),
          horizontalalignment='center',
          color=bar_color,
          weight='bold'
      )

    # Add labels and a title. Note the use of `labelpad` and `pad` to add some
    # extra space between the text and the tick labels.
    ax.set_xlabel(x_label, labelpad=15, color='#333333')
    ax.set_title(title, pad=15, color='#333333',
                 weight='bold')

    fig.tight_layout()

def create_matrix(df):
    '''Creates user-item matrix using scipy csr matrix
       
    Parameter:  df: dataframe: data
    '''
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
    
    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df['userId']), list(range(N))))
    movie_mapper = dict(zip(np.unique(df['movieId']), list(range(M))))
                        
    # Map indices to Ids
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df['userId'])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df['movieId'])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
    
    X = csr_matrix((df['rating'], (movie_index, user_index)), shape=(M, N))
                        
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
                        
def find_similar_movies(movie_id, df, k, metric='cosine', show_distance=False):
    '''Finds similar movies using KNN
       
    Parameter:  movie_id: int: movie's id
                df: dataframe: data
                k: int: number of neighbors  
    '''
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix( df )
    neighbor_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric)
    knn.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbor = knn.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbor.item(i)
        neighbor_ids.append(movie_inv_mapper[n])
    neighbor_ids.pop(0)
    return neighbor_ids


def main():
    '''main function
    Parameter:   None
    Return:      None
    '''
    #Read the data
    movies = pd.read_csv('movies.csv')
    movies.head()
    
    ratings = pd.read_csv('ratings.csv')
    ratings.head()

    imdb = pd.read_csv('IMDbMovies.csv')
    imdb = imdb[['Title', 'Release Year', 'Rating (Out of 10)']]
    imdb = imdb.dropna()
    imdb['Title'] = imdb['Title'].apply(lambda x: x[4:-1]+", The" if 'The' in x else x)
    imdb['Release Year'] = imdb['Release Year'].apply(lambda x: str(int(x)))
    imdb['Title'] = imdb[['Title','Release Year']].apply(lambda x: ' ('.join(x)+')', axis=1)
    imdb = imdb[['Title', 'Rating (Out of 10)']]
    imdb = imdb.rename( columns = {'Title': 'title', 'Rating (Out of 10)': 'Imdb Rating'})
    imdb.head()

    movies = pd.merge(movies, imdb, on='title', how='left', suffixes=('_restaurant_id', '_restaurant_review'))
    movies.head()

    #Explore Data
    n_ratings = len(ratings)
    n_movies = len(ratings['movieId'].unique())
    n_users = len(ratings['userId'].unique())

    max_rating = ratings['rating'].max()
    min_rating = ratings['rating'].min()

    print(f"Number of ratings: {n_ratings}")
    print(f"Number of movies: {n_movies}")
    print(f"Number of users: {n_users}")
    print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
    print(f"Average ratings per movie: {round(n_ratings/n_movies,2)}")
    print(f"Max rating: {max_rating}")
    print(f"Min rating: {min_rating}")

    movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
    movie_stats.columns = movie_stats.columns.droplevel()
    movie_stats.head()

    #Top 5 most rated movies (# of ratings)
    movie_stats_count = movie_stats.sort_values(['count'], ascending = False)[:5]

    movies_top5_count_titles = movies.loc[movies['movieId'].isin(movie_stats_count.index.values)]['title'].to_string(index=False)
    movies_top5_count_titles = movies_top5_count_titles.split('\n')
    rating_counts = movie_stats_count['count'].values

    plot_bars(movies_top5_count_titles, rating_counts, '# of ratings', 'Most rated movies')

    #Top 5 highest rated movies
    movie_stats_mean = movie_stats.loc[movie_stats['count']>=100].sort_values(['mean'], ascending = False)[:5]
    movies_top5_mean_titles = movies.loc[movies['movieId'].isin(movie_stats_mean.index.values)]['title'].to_string(index=False)
    movies_top5_mean_titles = movies_top5_mean_titles.split('\n')
    rating_mean = movie_stats_mean['mean'].values

    plot_bars(movies_top5_mean_titles, rating_mean, 'Rating', 'Highest rated movies' )

    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

    print(f"Please give a title of a movie you like")
    movie = 'Interstellar'
    titles = np.array(movies['title'].unique())
    res = [i for i in titles if movie.lower() in i.lower()]

    movie_title = ''
    if res:
        movie_title = res[0]

    movie_titles = dict(zip(movies['movieId'], movies['title']))
    movie_imdb = dict(zip(movies['movieId'], movies[ 'Imdb Rating' ]))

    if not movie_title:
        print(f"Please try another movie title")
    else:
        print(f"\033[1mSince you watched {movie_title}\033[0m")
        movie_id = movies.loc[movies['title']==movie_title]['movieId'].to_list()[0]
        similar_ids = find_similar_movies(movie_id, ratings, k=10)
        for i in similar_ids:
            if str(movie_imdb[i]) !='nan':
                print(movie_titles[i] + ' , Imdb: ' + str(movie_imdb[i]))
            else:
                print(movie_titles[i])
            
if __name__ == '__main__':
    main()