import numpy as np
import logging
class BaseMF:
    """
    Matrix Factorization Base Class
    """
    def __init__(self, M):
        
        self.train_errors = []
        self.val_errors = []
        self.M = M
        # M is the number of latent factors

    def initialize(self,num_unique_users, num_unique_items, M):
        """
        Initialize the latent factors and bias vectors. 
        Args:
            num_unique_users (int): number of users. Used as the first dimention of user matrix.
            num_unique_items (int): number of items. Used as the first dimention of items matrix.
            M (int): Number of latent factors.
        """
        self.user_latent_factor = np.random.random((num_unique_users, M))
        self.product_latent_factor = np.random.random((M, num_unique_items))
        self.user_bias = np.zeros((num_unique_users, 1))
        self.product_bias = np.zeros((num_unique_items,1))
        self.offset = 0

    def predict(self, u, p):
        """
        Predict the rating for a user and an item using their Ordinal Codes.
        Args:
            u (int): user code
            p (int): item code
        Returns:
            (int): estimated rating
        """
        return np.matmul(self.user_latent_factor[u,:],self.product_latent_factor[:,p]) + self.user_bias[u,0] + self.product_bias[p,0] + self.offset
    
    def calculate_mae(self, df):
        """
        Calculate Mean Absolute Error for provided df
        Args:
            df (pd.DataFrame): dataframe with columns same as 'user_col', 'item_col', and 'rating_col'.
        Returns:
            (float): mean absolute error of estimations.
        """

        if not self.user_col:
            logging.error("Please train the model.")

        errors = []
        for row in df.iterrows():
            u = int(row[1][self.user_col])
            p = int(row[1][self.item_col])
            
            pred = self.predict(u,p)
            error = row[1][self.rating_col] - pred
            errors.append(error)
        return np.mean(np.abs(errors))
    
    def train(self):
        """
        Train method.
        """
        pass

