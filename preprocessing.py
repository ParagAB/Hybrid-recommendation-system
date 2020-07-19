import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def preprocessing(movies,tags,ratings):
    
    movies['genres'] = movies['genres'].str.replace('|',' ')
    ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 55)

    # list the movie titles that survive the filtering
    movie_list_rating = ratings_f.movieId.unique().tolist()
    movies = movies[movies.movieId.isin(movie_list_rating)]
    Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    tags.drop(['timestamp'],1, inplace=True)
    ratings_f.drop(['timestamp'],1, inplace=True)

    mixed = pd.merge(movies, tags, on='movieId', how='left')

    mixed.fillna("", inplace=True)
    mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
                                          
    Final = pd.merge(movies, mixed, on='movieId', how='left')
    Final ['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis = 1)

    ratings_f1 = pd.merge(movies[['movieId']], ratings_f, on="movieId", how="right")
    ratings_f2 = ratings_f1.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)

    return (Final,ratings_f2)

def compress(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    tfdf = pd.DataFrame(tfidf_matrix.toarray(), index=df.index.tolist())
    return tfdf

def contentSVD(df,tfidf):
    svd = TruncatedSVD(n_components=200)
    latent_matrix = svd.fit_transform(tfidf)
    #number of latent dimensions to keep
    n = 200 
    latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index=df.title.tolist())

    latent_matrix_1_df.to_csv('LFeature1.csv')

def colabSVD(ratings,df):
    svd = TruncatedSVD(n_components=200)
    latent_matrix_2 = svd.fit_transform(ratings)
    latent_matrix_2_df = pd.DataFrame(latent_matrix_2,index=df.title.tolist())

    latent_matrix_2_df.to_csv('LFeature2.csv')

if __name__ == "__main__":
    #Download data from following source and make directory in the current directory called 'Dataset' 
    # and store files in the newly created directory
    #Datasource - https://www.kaggle.com/grouplens/movielens-20m-dataset/data?select=tag.csv
    try:
    
        movies = pd.read_csv('Dataset/movie.csv')
        tags = pd.read_csv('Dataset/tag.csv')
        ratings = pd.read_csv('Dataset/rating.csv')
        print('Files loaded Sucessfully')
    except:
        print('Error in loading files')
        sys.exit(1)
    
    try:
        content_df,Colab_df = preproecssing(movies,tags,ratings)
        print('Data Preprocessing successfull')
    except :
        print('Files could not be preprocessed')
        sys.exit(1)
    
    
    
    try:
        compressed_content_df = compress(content_df)
        contentSVD(content_df,compressed_content_df)
        print("LatentFeature for content filtering created successfully")
        Colab_df(Colab_df,content_df)
        print('LatentFeatures for Colaborative filtering created successfully')

    except:
        print("Latent features cannot be created")
        sys.exit(1)


