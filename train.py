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

    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="Batch size (default: 1024)")

    args = parser.parse_args()

    EPOCHS = args.epochs
    batch_size = args.batch_size

    log_info(f"{EPOCHS = }")
    log_info(f"{batch_size = }")

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
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)

    train_dataset = RecommendationDataset(train_df)
    test_dataset = RecommendationDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_countries = len(df_merged['user_country'].unique())
    max_age = int(df_merged['user_age'].max()) + 1
    num_publishers = len(df_merged['publisher'].unique())
    num_authors = len(df_merged['book_author'].unique())
    
    model = RecommendationModel(num_countries, max_age, num_publishers, num_authors)

    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    num_training_steps = len(train_loader) * EPOCHS  
    num_warmup_steps = 0.1 * num_training_steps 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    losses = []

    for epoch in range(EPOCHS):
        log_info(f"{epoch}/{EPOCHS} Running!!")
        running_loss = 0.0
        for batch_no, batch in enumerate(train_loader):
            user_age = torch.tensor(batch[0]).to(device)
            user_country = torch.tensor(batch[1]).to(device)
            year = torch.tensor(batch[2]).to(device)
            publisher = torch.tensor(batch[3]).to(device)
            author = torch.tensor(batch[4]).to(device)
            title = batch[5]
            rating = torch.tensor(batch[6]).to(device)
            
            optimizer.zero_grad()
            
            outputs = model(user_age, user_country, year, publisher, author, title)
            
            loss = criterion(outputs.squeeze(), rating)

            loss.backward()
            optimizer.step()
            scheduler.step()  

            loss_ = loss.item()
            losses.append(loss_)
            running_loss += loss_

            if batch_no % 1 == 0:
                log_info(f"Batch Finished {batch_no}!!! Loss : {loss_} lr : {scheduler.get_last_lr()}")
                
        log_info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        torch.save(model.state_dict(), f'models/recommendation_model_{epoch}.pth')

    torch.save(model.state_dict(), 'models/recommendation_model.pth')

    log_info(f"Model Trained Successfully. Exiting with Code 0")