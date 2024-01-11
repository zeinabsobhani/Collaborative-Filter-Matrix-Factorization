"""
matrix factorization with stochastic gradient descent
"""
import numpy as np
import yaml

with open("./config/config.yaml", "rt") as config_file:
    config = yaml.safe_load(config_file)

class SGD_MF:
    def __init__(self, M):

        self.train_errors = []
        self.val_errors = []
        self.M = M

    def random_initialize(self,num_unique_users, num_unique_items, M):
        self.user_latent_factor = np.random.random((num_unique_users, M))
        self.product_latent_factor = np.random.random((M, num_unique_items))
        self.user_bias = np.random.random((num_unique_users, 1))
        self.product_bias = np.random.random((num_unique_items,1))
        self.offset = 0

    def calculate_mae(self, df):
        errors = []
        for row in df.iterrows():
            u = int(row[1][self.user_col])
            p = int(row[1][self.item_col])
            
            pred = self.predict(u,p)
            error = row[1][self.rating_col] - pred
            errors.append(error)
        return np.mean(np.abs(errors))

    def train(self, df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings', n_epoch = 10, lr = 0.1 , lambda_ub = 0.1, lambda_pb = 0.1, lambda_u = 0.1, lambda_p = 0.1, lambda_offset = 0.1):

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        # different initialization for offset
        num_unique_users = df_train[user_col].nunique()
        num_unique_items = df_train[item_col].nunique()

        self.random_initialize(num_unique_users, num_unique_items, self.M)
        self.offset = np.mean(df_train[rating_col])
    
        for epoch in range(n_epoch):
            print(f"epoch: {epoch}")
            errors = []
            for row in df_train.iterrows():
                u = int(row[1][user_col])
                p = int(row[1][item_col])
                
                pred = self.predict(u,p)
                error = row[1][rating_col] - pred

                self.user_bias[u] += lr *( error - lambda_ub * self.user_bias[u])
                self.product_bias[p] += lr * (error - lambda_pb * self.product_bias[p])

                self.offset += lr * (error - lambda_offset * self.offset)
                self.user_latent_factor[u,:] = self.user_latent_factor[u,:] + lr* (
                    error * self.product_latent_factor[:,p] - lambda_u*self.user_latent_factor[u,:]
                )
                self.product_latent_factor[:,p] = self.product_latent_factor[:,p] + lr* (
                    error * self.user_latent_factor[u,:] -lambda_p*self.product_latent_factor[:,p]
                )
                errors.append(error)

            train_mae = np.mean(np.abs(errors))
            print("train MAE error: ",train_mae)
            val_error = self.calculate_mae(df_val)
            print("val MAE error: ",val_error)
            self.train_errors.append(train_mae)
            self.val_errors.append(val_error)



    def predict(self, u, p):
        return np.matmul(self.user_latent_factor[u,:],self.product_latent_factor[:,p]) + self.user_bias[u] + self.product_bias[p] + self.offset



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
mf = SGD_MF(M)

# print(**config['sgd_params'])

mf.train(df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings',**config['sgd_params'])