# Collaborative-Filter-Matrix-Factorization

In this repository, you can find 3 different method of calculating Latent Factors ($P$ and $Q$) to be used for a recommender system.
1. Stochastic Gradient Descent
2. Alternating Least Square Method
3. Embeddings

The last method utilizes Pytorch to train an encoder for users and items instead of directly calculating the matrices.
The dataset used as part of this, is the small [Amazon product rating](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) dataset for the *Arts_Crafts_and_Sewing* category. 

## Prerequisites

Python 3.11.6

Please install `requirements.txt`. 
- Pandas
- Numpy
- sklearn
- pytorch


## How to Use

You can use `config/config.yaml` file to set your custom parameters for each method. 

### Data download
Similarly, changing `URL` parameter to point to a different category would change your data source. Alternatively, you can pass `url` argument directly to the function.
To populate dataset in a json format in `data/raw/` directory, run the following: 
```
from download_data import DataLoader
dl = DataLoader()
dl.download_and_extract_from_url(url = 'https://xxxxxxx/Arts_Crafts_and_Sewing_5.json.gz', save_to = 'Arts_Crafts_and_Sewing_5.json')
``` 

To load all the data from `data/raw/` directory in a pandas dataframe: 
```
df = dl.load_raw_as_df()
```

### Preprocessing
For the preprocessor, you need to define the columns to be considered as *User*, *Item* and *Time*. 
`full_preprocess` implements the following: 
1. Parse dates columns
2. Remove duplicate entries
3. Filter users with less than *num_tests* number of ratings.
4. Split dataset into test and train. The split is done based on the time. The last *num_tests* number of user reviews are set aside for the test set. This is to keep the time based nature of the dataset. 
5. Encode user and item columns with and OrdinalEncoder.

```
pp = PreProcessor(df, user_col = 'reviewerID',item_col = 'asin', time_col = 'reviewTime')
pp.full_preprocess(num_tests=2)
```

### SGD and ALS
to train either of these methods: 
```
from download_data import DataLoader
from preprocessor import PreProcessor
from als_mf import ALS_MF
from sgd_mf import SGD_MF

dl = DataLoader()
df = dl.load_raw_as_df()
df = df.rename(columns = {'overall':'ratings'})
pp = PreProcessor(df, user_col = 'reviewerID',item_col = 'asin', time_col = 'reviewTime')
pp.full_preprocess(num_tests=2)

df_train = pp.df_train
df_val = pp.df_val

M = 16
mf = ALS_MF(M)
mf.train(df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings',**config['als_params'])
mf.predict(u =10, p = 10)


mf = SGD_MF(M)
mf.train(df_train , df_val, user_col = 'reviewerID', item_col = 'asin', rating_col = 'ratings',**config['als_params'])
mf.predict(u =10, p = 10)
```
