import numpy as np
from numpy.linalg import solve
import yaml

with open("./config/config.yaml", "rt") as config_file:
    config = yaml.safe_load(config_file)

class ALS_MF:
    def __init__(self, M):
        
        self.train_errors = []
        self.val_errors = []
        self.M = M

    def random_initialize(self,num_unique_users, num_unique_items, M):
        self.user_latent_factor = np.random.random((num_unique_users, M))
        self.product_latent_factor = np.random.random((M, num_unique_items))


    def calculate_mae(self, df):
        errors = []
        for row in df.iterrows():
            u = int(row[1][self.user_col])
            p = int(row[1][self.item_col])
            
            pred = self.predict(u,p)
            error = row[1][self.rating_col] - pred
            errors.append(error)
        return np.mean(np.abs(errors))
    

    def predict(self, u, p):
        return np.matmul(self.user_latent_factor[u,:],self.product_latent_factor[:,p])
    
    def pivot_target(self, df):
        rows, row_pos = np.unique(df[self.user_col], return_inverse=True)
        cols, col_pos = np.unique(df[self.item_col], return_inverse=True)
        
        pivot_table = np.zeros((len(rows), len(cols)))
        pivot_table[row_pos, col_pos] = df[self.rating_col]

        return pivot_table

    def train(self, df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings', n_epoch = 10, lambda_u = 0.1, lambda_p = 0.1,):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        num_unique_users = df_train[user_col].nunique()
        num_unique_items = df_train[item_col].nunique()

        self.random_initialize(num_unique_users, num_unique_items, self.M)
        ratings = self.pivot_target(df_train)

        # excluding 0 ratings from the training
        w = (ratings>0).astype(int)

        for epoch in range(n_epoch):
            print(f"epoch: {epoch}")
            
            for u in range(num_unique_users):
                
                 
                XTX = (self.product_latent_factor * (w[u,:].reshape(1,-1))).dot(self.product_latent_factor.T)
                lambdaI = np.eye(self.M) * lambda_u
            
                self.user_latent_factor[u, :] = solve((XTX + lambdaI), 
                                            (ratings[u, :]*w[u,:]).dot(self.product_latent_factor.T ))

            

            for p in range(num_unique_items):
                
                YTY = (self.user_latent_factor.T*w[:,p]).dot(self.user_latent_factor)
                lambdaI = np.eye(self.M) * lambda_p
            
                self.product_latent_factor[:, p] = solve((YTY + lambdaI), 
                                            (ratings[:, p]*w[:,p]).dot(self.user_latent_factor))

            
            train_error = self.calculate_mae(df_train,  self.user_latent_factor, self.product_latent_factor)
            val_error = self.calculate_mae(df_val,  self.user_latent_factor, self.product_latent_factor)

            print("train error: ", train_error)
            print("val error: ", val_error)


from download_data import DataLoader
from preprocessor import PreProcessor
dl = DataLoader()
df = dl.load_raw_as_df()
print(df.head())
df = df.rename(columns = {'overall':'ratings'})
pp = PreProcessor(df, user_col = 'reviewerID',item_col = 'asin', time_col = 'reviewTime')
pp.full_preprocess(num_tests=2)

df_train = pp.df_train
df_val = pp.df_val

M = 16
mf = ALS_MF(M)

# print(**config['sgd_params'])

mf.train(df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings',**config['als_params'])