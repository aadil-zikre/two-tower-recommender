import pandas as pd
import numpy as np
from krezi import *
from .utils import log_file_init_name

init_logger(log_file_init_name())

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from .utils import preprocess_users, preprocess_books, preprocess_ratings, consolidate_dfs
from .models import UserTower, ItemTower, RecommendationModel
from .dataloader import RecommendationDataset, collate_fn

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference Script")

    # Required arguments
    parser.add_argument("-a", "--user_age", type=int, required=True, help="User's age")
    parser.add_argument("-c", "--user_country", type=int, required=True, help="User's country (integer code)")
    parser.add_argument("-i", "--user_id", type=int, required=True, help="User's ID")

    # Optional argument with default value
    parser.add_argument("--model", default="recommendation_model_2.pth", help="Name to the model saved in models/ dir (default: recommendation_model_2.pth)")

    args = parser.parse_args()

    # Access the parsed arguments
    user_age = args.user_age
    user_country = args.user_country
    user_id = args.user_id
    model_path = args.model

    df_users = pd.read_csv("data/Users.csv")
    df_books = pd.read_csv("data/Books.csv")
    df_ratings = pd.read_csv("data/Ratings.csv")

    df_users = preprocess_users(df_users)
    df_books = preprocess_books(df_books)
    df_ratings = preprocess_ratings(df_ratings, df_users, df_books)

    df_merged = consolidate_dfs(df_ratings, df_users, df_books)

    scalar_year = MinMaxScaler(feature_range=(-1, 1))
    scalar_age = MinMaxScaler(feature_range=(-1, 1))

    df_merged['year_of_publication'] = scalar_year.fit_transform(df_merged['year_of_publication'].values.reshape(-1,1))
    df_merged['user_age'] = scalar_age.fit_transform(df_merged['user_age'].values.reshape(-1,1))
    
    le_book_author = LabelEncoder()
    le_publisher = LabelEncoder()
    le_user_country = LabelEncoder()

    df_merged['book_author'] = le_book_author.fit_transform(df_merged['book_author'])
    df_merged['publisher'] = le_publisher.fit_transform(df_merged['publisher'])
    df_merged['user_country'] = le_user_country.fit_transform(df_merged['user_country'])
    
    num_countries = len(df_merged['user_country'].unique())
    max_age = int(df_merged['user_age'].max()) + 1
    num_publishers = len(df_merged['publisher'].unique())
    num_authors = len(df_merged['book_author'].unique())
    
    model = RecommendationModel(num_countries, max_age, num_publishers, num_authors)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    try:
        model.load_state_dict(torch.load(f'models/{model_path}', map_location=device))
    except:
        log_info("Error Fetching Saved Model. Check if the model exists in models/")
        log_info("If not found, run train.py script first.")
        sys.exit(404)

    model.eval()
    with torch.no_grad():
        df_pred = df_merged.copy()
        df_pred = df_pred.drop_duplicates(subset='isbn', ignore_index=True)
        df_pred['user_id'] = user_id
        df_pred['user_age'] = user_age
        df_pred['user_age'] = scalar_age.transform(df_pred['user_age'].values.reshape(-1,1))
        df_pred['user_country'] = user_country
        df_pred['rating'] = 0
        inference_set = RecommendationDataset(df_pred)
        inference_loader = DataLoader(inference_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print("Inference Batches :: ", len(inference_loader))
        outputs_list = []
        for batch_no, batch in enumerate(inference_loader):
            user_age = torch.tensor(batch[0]).to(device)
            user_country = torch.tensor(batch[1]).to(device)
            year = torch.tensor(batch[2]).to(device)
            publisher = torch.tensor(batch[3]).to(device)
            author = torch.tensor(batch[4]).to(device)
            # title = [tokenizer.encode(title, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device) for title in batch[5]]
            title = batch[5]
            rating = torch.tensor(batch[6]).to(device)
            with torch.no_grad():
                outputs = model(user_age, user_country, year, publisher, author, title)
            print(outputs.squeeze())
            outputs_list.append(outputs.squeeze().cpu().numpy())
        outputs_list = np.concatenate(outputs)
        df_pred['rating'] = outputs_list
        df_pred = df_pred.sort_values(by='rating', ascending=False)
        print(df_pred.head(10))

    log_info(f"Inference Successful!")