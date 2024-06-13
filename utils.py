import pandas as pd
import numpy as np
from krezi import *
import os 
import datetime

def preprocess_users(df_users):
    
    df_users.Age = df_users.Age.clip(
                                lower=df_users.Age.quantile(0.05),
                                upper=df_users.Age.quantile(0.95)
                                )
    
    df_users.Age = df_users.Age.fillna(int(df_users.Age.mean()))
    
    df_users['country'] = df_users['Location'].str.rsplit(",", n=1, expand=True).iloc[:,-1].str.strip()

    df_users = df_users[df_users.country.isin(['usa', 'canada', 'united kingdom'])]

    return df_users

def preprocess_books(df_books):
    
    df_books = df_books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]

    df_books['Year-Of-Publication'] = pd.to_numeric(df_books['Year-Of-Publication'], errors='coerce')

    df_books = df_books.dropna(subset=['Year-Of-Publication'])

    df_books.loc[df_books['Year-Of-Publication'] == 0] = 1376 # cant be 0, replacing with minimum
    df_books.loc[df_books['Year-Of-Publication'] > 2024] = 2024

    return df_books

def preprocess_ratings(df_ratings, df_users, df_books):
    
    valid_user_ids = set(df_users['User-ID'].values)

    df_ratings = df_ratings[df_ratings['User-ID'].apply(lambda id : True if id in valid_user_ids else False)]

    valid_isbns = set(df_books.ISBN.values)

    df_ratings = df_ratings[df_ratings['ISBN'].apply(lambda isbn : True if isbn in valid_isbns else False)]

    return df_ratings 

def consolidate_dfs(df_ratings, df_users, df_books):

    df_users.columns = ['user_id', 'user_location', 'user_age', 'user_country']
    df_users = df_users[['user_id', 'user_age', 'user_country']]

    df_books.columns = ['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher']
    
    df_ratings.columns = ['user_id', 'isbn', 'rating']

    df_merged = df_ratings.merge(df_books, on='isbn')
    df_merged = df_merged.merge(df_users, on='user_id')

    df_merged.dropna(how='any', ignore_index=True, inplace=True)

    df_merged['rating'] = df_merged['rating'].astype(float)

    df_merged['user_age'] = df_merged['user_age'].astype(float)

    return df_merged


def log_file_init_name():
    log_files = sorted([f for f in os.listdir("logs") if f.endswith(".log")])
    if log_files:
        last_log_file = log_files[-1]
        log_file_number = int(last_log_file.split(".")[0].split("_")[-2]) + 1
    else:
        log_file_number = 1
    current_time = datetime.now().strftime("%Y%m%d:%H%M%S")
    log_file_name = f"two_tower_recommender_ver_{log_file_number}_{current_time}.log"
    log_file_path = os.path.join("logs", log_file_name)
    return log_file_path