"""
matrix factorization with stochastic gradient descent
"""
import numpy as np
import yaml
from base_mf import BaseMF

with open("./config/config.yaml", "rt") as config_file:
    config = yaml.safe_load(config_file)

class SGD_MF(BaseMF):
    """
    Matrix Factorization with SGD Class. Inherits the Base Matrix Factorization Class.
    """
    def train(self, df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings', n_epoch = 10, lr = 0.1 , lambda_ub = 0.1, lambda_pb = 0.1, lambda_u = 0.1, lambda_p = 0.1, lambda_offset = 0.1):
        """
        SGD training. Initializes and updates attributes `user_latent_factor`, `product_latent_factor`, `user_bias`, `product_bias`, `offset`,
                    `train_errors` and `val_errors`.
        Args:
            df_train (pd.DataFrame): training data set
            df_val (pd.DataFrame): validation dataset
            user_col (str): name of the column to be used as user, Default `reviewerID`.
            item_col (str): name of the column to be used as item, Default `asin`.
            rating_col (str): name of the column to be used as rating, Default `ratings`.
            n_epoch (int): number of training epochs, Default 10.
            lr (float): Learning rate, Default 0.1.
            lambda_ub (float): user bias vector regularization, Default 0.1.
            lambda_pb (float): item bias vector regularization, Default 0.1.
            lambda_u (float): user matrix regularization, Default 0.1.
            lambda_p (float): item matrix regularization, Default 0.1.
            lambda_offset (float): offset (global bias) regularization, Default 0.1.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        
        num_unique_users = df_train[user_col].nunique()
        num_unique_items = df_train[item_col].nunique()

        self.initialize(num_unique_users, num_unique_items, self.M)
        # different initialization for offset
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

