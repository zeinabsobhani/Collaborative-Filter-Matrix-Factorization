import pandas as pd
import logging
from sklearn.preprocessing import OrdinalEncoder
import logging


class PreProcessor:
    """
    The Preprocessor class handles data cleaning and manipulation to prepare data for Matrix Factorization
    """
    def __init__(self, df, user_col = 'reviewerID',item_col = 'asin', time_col = 'reviewTime'):
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.df = df


    def parse_dates(self, date_cols):
        """
        Parse str date columns into datetime
        """
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col])

    def remove_duplicates(self):
        """
        Remove duplicate entries. Rows with same user and item are considered duplicates. 
        """
        self.df = self.df.set_index([self.user_col, self.item_col, self.time_col ]).sort_index()
        self.df = self.df.reset_index()
        self.df = self.df [~ self.df[[self.user_col, self.item_col,]].duplicated(keep = 'last')]

    def filter_min_ratings_count(self, by = 'user', min_ratings = 2):
        """
        Filter the instances where user or item has less than min_ratings rows. 
        Args:
            by (str): `user` or `item`. If `user`, filters users with less than `n` ratings.
                        If `item`, filters items less than `n` ratings.
            min_ratings (int): threshold to filter the rating counts by. 
        """
        if by == 'user':
            self.df = self.df.groupby(self.user_col).filter(lambda x: len(x) > min_ratings) 
        elif by=='item':
            self.df = self.df.groupby(self.item_col).filter(lambda x: len(x) > min_ratings) 
        else:
            logging.error("invalid input for filter_min_ratings_count by, it should be 'user' or 'item'")

    def test_train_split(self, num_tests = 2):
        """
        Split data into test and train. This split is time-based to save the nature of the application.
        Select last `num_tests` ratings of each user as test.
        Args:
            num_tests (int): number of ratings to set in test dataset
        Returns: 
            df_train, df_val (pd.DataFrame): training and validation datasets
        """
        df_train = self.df.groupby(self.user_col).head(-num_tests).reset_index(drop=True)
        df_val = self.df.groupby(self.user_col).tail(num_tests).reset_index(drop=True)

        # make sure the items in test exist in training as well
        df_val = df_val[df_val[self.item_col].isin(df_train[self.item_col])]

        logging.info(f"{df_train.shape[0]} training samples")
        logging.info(f"{df_val.shape[0]} test samples")

        self.df_train = df_train
        self.df_val = df_val

        return df_train,df_val
    
    def encode(self, cols_to_encode):
        """
        Uses OrdinalEncoder to encode user and items.
        Args:
            cols_to_encode (List): List of columns to encode.
        Returns: 
            df_train, df_val (pd.DataFrame): encoded training and validation datasets
        """
        oe = OrdinalEncoder()
        oe.fit(self.df_train[cols_to_encode])
        self.df_train[cols_to_encode]=oe.transform(self.df_train[cols_to_encode])
        self.df_val[cols_to_encode]=oe.transform(self.df_val[cols_to_encode])

        self.df_train[cols_to_encode] = self.df_train[cols_to_encode].astype(int)
        self.df_val[cols_to_encode] = self.df_val[cols_to_encode].astype(int)

    
        return self.df_train, self.df_val
    
    def full_preprocess(self, num_tests = 2):
        """
        Performs all the preprocessing steps.
        """
        self.parse_dates([self.time_col])
        self.remove_duplicates()
        self.filter_min_ratings_count(by = 'user', min_ratings = num_tests)
        self.test_train_split(num_tests = num_tests)
        self.encode([self.user_col, self.item_col])

