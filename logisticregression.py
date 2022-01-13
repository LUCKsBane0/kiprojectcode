import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iterations = 25

hearts = pd.read_csv('hvectors.csv')
triangles = pd.read_csv('dvectors.csv')
circles = pd.read_csv('kvectors.csv')
squares = pd.read_csv('qvectors.csv')

hearts.drop(hearts.columns[0], axis=1, inplace=True)
triangles.drop(triangles.columns[0], axis=1, inplace=True)
circles.drop(circles.columns[0], axis=1, inplace=True)
squares.drop(squares.columns[0], axis=1, inplace=True)

def split_data_set(df_to_be_split: pd.DataFrame):
    holy_numbers = [i for i in range(0, len(df_to_be_split))]
    random.shuffle(holy_numbers)
    train_numbers = holy_numbers[:int(0.80*len(df_to_be_split))]
    test_numbers = holy_numbers[int(0.80*len(df_to_be_split)):int(1.00*len(df_to_be_split))]
    train_set = df_to_be_split.iloc[train_numbers]
    test_set = df_to_be_split.iloc[test_numbers]
    return train_set, test_set
hearts['2500'] = 0
triangles['2500'] = 1
circles['2500'] = 2
squares['2500'] = 3
