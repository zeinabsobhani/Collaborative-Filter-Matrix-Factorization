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
