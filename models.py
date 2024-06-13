import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
bert_model.eval()

EMBEDDING_DIM = 8

# User Tower
class UserTower(nn.Module):
    def __init__(self, num_countries):
        super(UserTower, self).__init__()
        self.user_country_embed = nn.Embedding(num_countries, 8)
        self.fc1 = nn.Linear(9, EMBEDDING_DIM)
        self.double()

    def forward(self, user_age, user_country):
        age_input = user_age.unsqueeze(1)
        country_embed = self.user_country_embed(user_country)
        x = torch.cat([age_input, country_embed], dim=1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        return x

# Item Tower
class ItemTower(nn.Module):
    def __init__(self, num_publishers, num_authors):
        super(ItemTower, self).__init__()
        self.publisher_embed = nn.Embedding(num_publishers, 32)
        self.author_embed = nn.Embedding(num_authors, 32)
        self.lstm_bert_1 = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.lrelu_lstm_bert_1 = nn.LeakyReLU(0.2)
        self.bert_fc1 = nn.Linear(256*2, 128)
        self.lrelu_bert_1 = nn.LeakyReLU(0.2)
        self.bert_fc2 = nn.Linear(128, 64)
        self.lrelu_bert_2 = nn.LeakyReLU(0.2)
        self.bert_fc3 = nn.Linear(64, 32)
        self.all_fc1 = nn.Linear(97, EMBEDDING_DIM)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.double()

    def forward(self, year, publisher, author, title):
        year_input = year.unsqueeze(1)
        publisher_embed = self.publisher_embed(publisher)
        author_embed = self.author_embed(author)

        title_ids = tokenizer.batch_encode_plus(title, pad_to_max_length=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            title_output = bert_model(**title_ids)[1]
        title_output = title_output.double()
        title_output, _ = self.lstm_bert_1(title_output)
        title_output = self.lrelu_lstm_bert_1(title_output)
        title_output = self.bert_fc1(title_output)
        title_output = self.lrelu_bert_1(title_output)
        title_output = self.bert_fc2(title_output)
        title_output = self.lrelu_bert_2(title_output)
        title_output = self.bert_fc3(title_output)
        title_output = torch.nn.functional.relu(title_output)
        x = torch.cat([year_input, publisher_embed, author_embed, title_output], dim=1)
        x = self.all_fc1(x)
        x = torch.nn.functional.relu(x)
        return x

# Combined Model
class RecommendationModel(nn.Module):
    def __init__(self, num_countries, max_age, num_publishers, num_authors):
        super(RecommendationModel, self).__init__()
        self.user_tower = UserTower(num_countries)
        self.item_tower = ItemTower(num_publishers, num_authors)
        self.user_embedding_dim = 1 + 32
        self.item_embedding_dim =  1 + 32 + 32 + 768

    def forward(self, user_age, user_country, year, publisher, author, title):
        user_tower_output = self.user_tower(user_age, user_country)
        item_tower_output = self.item_tower(year, publisher, author, title)
        user_embeddings = user_tower_output.view(-1, 1, EMBEDDING_DIM)
        item_embeddings = item_tower_output.view(-1, EMBEDDING_DIM, 1)
        scores = torch.bmm(user_embeddings, item_embeddings)
        return scores