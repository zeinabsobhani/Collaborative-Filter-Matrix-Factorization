import numpy as np
from numpy.linalg import solve
import yaml
from base_mf import BaseMF

with open("./config/config.yaml", "rt") as config_file:
    config = yaml.safe_load(config_file)

class ALS_MF(BaseMF):
    """
    Matrix Factorization with ALS Class. Inherits the Base Matrix Factorization Class.
    """
    def pivot_target(self, df):
        """
        Creates matrix of ratings where rows are users and columns are items, by pivoting. For easier query in training.
        Args:
            df (pd.DataFrame): dataframe to pivot.
        Returns:
            pivot_table (np.array): matrix of ratings
        """
        rows, row_pos = np.unique(df[self.user_col], return_inverse=True)
        cols, col_pos = np.unique(df[self.item_col], return_inverse=True)
        
        pivot_table = np.zeros((len(rows), len(cols)))
        pivot_table[row_pos, col_pos] = df[self.rating_col]

        return pivot_table

    def train(self, df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings', n_epoch = 10, lambda_u = 0.1, lambda_p = 0.1,):
        """
        ALS training. Initializes and updates attributes `user_latent_factor`, `product_latent_factor`,
                    `train_errors` and `val_errors`.
        Args:
            df_train (pd.DataFrame): training data set
            df_val (pd.DataFrame): validation dataset
            user_col (str): name of the column to be used as user, Default `reviewerID`.
            item_col (str): name of the column to be used as item, Default `asin`.
            rating_col (str): name of the column to be used as rating, Default `ratings`.
            n_epoch (int): number of training epochs, Default 10.
            lambda_u (float): user matrix regularization, Default 0.1.
            lambda_p (float): item matrix regularization, Default 0.1.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        num_unique_users = df_train[user_col].nunique()
        num_unique_items = df_train[item_col].nunique()

        self.initialize(num_unique_users, num_unique_items, self.M)
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

            
            train_error = self.calculate_mae(df_train)
            val_error = self.calculate_mae(df_val)

            print("train MAE error: ", train_error)
            print("val MAE error: ", val_error)

            self.train_errors.append(train_error)
            self.val_errors.append(val_error)
