import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier

iterations = 25

hearts = pd.read_csv('hvectors.csv')
triangles = pd.read_csv('dvectors.csv')
circles = pd.read_csv('kvectors.csv')
squares = pd.read_csv('qvectors.csv')

hearts.drop(hearts.columns[0], axis=1, inplace=True)
triangles.drop(triangles.columns[0], axis=1, inplace=True)
circles.drop(circles.columns[0], axis=1, inplace=True)
squares.drop(squares.columns[0], axis=1, inplace=True)

hearts['2500'] = 0
triangles['2500'] = 1
circles['2500'] = 2
squares['2500'] = 3

k_sum = 0
error_sum = 0

k_number_used = 10 * [0]

def k_nearest_neighbors(k_neighbors, training_set):
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
    classes = training_set.iloc[: , len(training_set.columns)-1]
    #print(classes)
    knn.fit(training_set.drop(training_set.columns[-1], axis=1), classes)
    return knn

def test_error(knn, test_set):
    error = 1 - knn.score(test_set.drop(test_set.columns[-1], axis=1), test_set.iloc[: , -1])
    #print(error)
    return error

def test_knn(k, training_set, test_set):
    knn = k_nearest_neighbors(k, training_set)
    error = test_error(knn, test_set)
    return error

def get_best_k(training_set, test_set):
    k_list = [i for i in range(1, 11)]
    best_k = 0
    best_error = 1
    for k in k_list:
        error = test_knn(k, training_set, test_set)
        if (error < best_error):
            best_error = error
            best_k = k
    return best_k, best_error

def get_folds(data):
    folds = [data[0:int(0.2*len(data))], data[int(0.2*len(data)):int(0.4*len(data))], data[int(0.4*len(data)):int(0.6*len(data))], data[int(0.6*len(data)):int(0.8*len(data))], data[int(0.8*len(data)):int(1.0*len(data))]]
    return folds

def cross_validation(data):
    best_k = None
    best_error = 10
    folds = get_folds(data)
    for fold in folds:
        train_folds = []
        for f in folds:
            #print(len(f), f)
            if f is not fold:
                train_folds.append(f)
        train_set = pd.concat(train_folds)
        #print(train_set)
        test_set = fold
        current_k, current_error = get_best_k(train_set, test_set)
        if current_error < best_error:
            best_k = current_k
            best_error = current_error
    k_number_used[best_k-1] += 1
    return best_k, best_error

def split_data_set(df_to_be_split: pd.DataFrame):
    holy_numbers = [i for i in range(0, len(df_to_be_split))]
    random.shuffle(holy_numbers)
    train_numbers = holy_numbers[:int(0.80*len(df_to_be_split))]
    test_numbers = holy_numbers[int(0.80*len(df_to_be_split)):int(1.00*len(df_to_be_split))]
    train_set = df_to_be_split.iloc[train_numbers]
    test_set = df_to_be_split.iloc[test_numbers]
    return train_set, test_set

# find best_k with cross validation
for i in range(iterations):
    data = [hearts, triangles, circles, squares]
    data_set = pd.concat(data)
    data_set = data_set.sample(frac=1)

    k, error = cross_validation(data_set)
    k_sum += k
    error_sum += error

print(k_number_used)
print(k_sum/iterations, error_sum/iterations)

error_sum = 0
best_k = 0
times_used = 0
for i in range(0, 9):
    if k_number_used[i] > times_used:
        best_k = i+1
        times_used = k_number_used[i]

# train final model on best_k found
for i in range(iterations):
    hearts_train, hearts_test = split_data_set(hearts)
    triangles_train, triangles_test = split_data_set(triangles)
    circles_train, circles_test = split_data_set(circles)
    squares_train, squares_test = split_data_set(squares)

    train_set = [hearts_train, triangles_train, circles_train, squares_train]
    data_train = pd.concat(train_set)
    data_train = data_train.sample(frac=1)

    test_set = [hearts_test, triangles_test, circles_test, squares_test]
    data_test = pd.concat(test_set)
    data_test = data_test.sample(frac=1)

    error = test_knn(best_k, data_train, data_test)
    error_sum += error

print(best_k, error_sum/iterations)