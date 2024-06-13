
import torch
class RecommendationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (row['user_age'], row['user_country'], row['rating'],
                row['year_of_publication'], row['publisher'], row['book_author'], row['book_title'])

# Custom collate function
def collate_fn(batch):
    user_age = torch.tensor([item[0] for item in batch])
    user_country = [item[1] for item in batch]
    rating = torch.tensor([item[2] for item in batch])
    year = torch.tensor([item[3] for item in batch])
    publisher = [item[4] for item in batch]
    author = [item[5] for item in batch]
    title = [item[6] for item in batch]

    return user_age, user_country, year, publisher, author, title, rating