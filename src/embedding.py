import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
import torch.utils.data as tud
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("./config/config.yaml", "rt") as config_file:
    config = yaml.safe_load(config_file)

class RatingsDataset(Dataset):
    """
    Convert data into a format usable by pytorch Dataset module.
    """
    def __init__(self, df, user_col, item_col, rating_col):
        super().__init__()
        self.df = df[[user_col, item_col, rating_col]]
        self.x_user_item = list(zip(df[user_col].values, df[item_col].values))
        self.y_rating = self.df[rating_col].values
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.x_user_item[idx], self.y_rating[idx]
    

class MFAdvanced(nn.Module):
    """ Create the embedding architecture"""
    def __init__(self, num_users, num_items, M):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, M)
        self.item_emb = nn.Embedding(num_items, M)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        element_product = (user_emb*item_emb).sum(1)
        user_b = self.user_bias[user]
        item_b = self.item_bias[item]
        element_product += user_b + item_b + self.offset
        return element_product

class Embeddings:
    """ Class to train the model """
    def __init__(self, M , num_unique_users, num_unique_items ):
        self.M = M
        self.num_unique_users = num_unique_users
        self.num_unique_items = num_unique_items
        self.model = MFAdvanced(self.num_unique_users, self.num_unique_items, M=self.M)
        self.model.to(device)
        self.user_col, self.item_col, self.rating_col = None , None, None
        self.epoch_train_losses, self.epoch_val_losses = [], []

    
    def load_data(self, df_train, df_val, user_col, item_col, rating_col, batch = 1000,):
        """ 
        Convert datasets into a pytorch DataLoader object
        Args:
            df_train (pd.DataFrame): training dataset
            df_val (pd.DataFrame): validation dataset
            user_col (str): name of the column to be used as user, Default `reviewerID`.
            item_col (str): name of the column to be used as item, Default `asin`.
            rating_col (str): name of the column to be used as rating, Default `ratings`.
            batch (int): number of instances in each data batch.
        Returns:
            dl_train, dl_val (DataLoader): pytorch DataLoader objects
        """
        self.user_col, self.item_col, self.rating_col = user_col, item_col, rating_col

        ds_train = RatingsDataset(df_train, user_col, item_col, rating_col)
        ds_val =  RatingsDataset(df_val, user_col, item_col, rating_col)

        dl_train = tud.DataLoader(ds_train, batch, shuffle=True,) 
        dl_val = tud.DataLoader(ds_val, batch, shuffle=True)
        return dl_train, dl_val

    def train(self, dl_train , dl_val ,lr: 0.01, weight_decay = 0.01 , n_epoch = 20):
        """
        Train the embedding model.
        Args:
            dl_train (DataLoader): train dataloader
            dl_val (DataLoader): validation dataloader
            lr (float): Learning rate, Default 0.01.
            weight_decay (float): Form of regularization in AdamW Optimization, Default to 0.01.
            n_epoch (int): number of training epochs, Default 20.
        """
        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        
        for i in range(n_epoch):
            print(f"epoch: {i}")
            train_losses, val_losses = [], []
            self.model.train()
            for xb,yb in dl_train:
                xUser = xb[0].to(device, dtype=torch.long)
                xItem = xb[1].to(device, dtype=torch.long)
                yRatings = yb.to(device, dtype=torch.float)
                preds = self.model(xUser, xItem)
                loss = loss_fn(preds, yRatings)
                train_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            lpreds, lratings = [], []
            self.model.eval()
            for xb,yb in dl_val:
                xUser = xb[0].to(device, dtype=torch.long)
                xItem = xb[1].to(device, dtype=torch.long)
                yRatings = yb.to(device, dtype=torch.float)
                preds = self.model(xUser, xItem)
                loss = loss_fn(preds, yRatings)
                val_losses.append(loss.item())
                
                lpreds.extend(preds.detach().cpu().numpy().tolist())
                lratings.extend(yRatings.detach().cpu().numpy().tolist())

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            self.epoch_train_losses.append(epoch_train_loss)
            self.epoch_val_losses.append(epoch_val_loss)

            print("train MSE loss: ", epoch_train_loss)
            print("val MSE loss: ", epoch_val_loss)

