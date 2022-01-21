import pandas as pd
import random
from sklearn.linear_model import LogisticRegression

iterations = 1000

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

penalty_sum = 0
error_sum = 0

penalty_number_used = 4 * [0]

def penalty_logr(penalty, training_set):
    pen = ''
    if penalty == 0:
        pen = 'none'
    if penalty == 1:
        pen = 'l2'
    if penalty == 2:
        pen = 'l1'
    if penalty == 3:
        pen = 'l2'
    logr = LogisticRegression(penalty=pen,
                            multi_class='auto',
                            solver='saga',
                            max_iter=100,
                            tol=1e-2)
    classes = training_set.iloc[: , len(training_set.columns)-1]
    #print(classes)
    logr.fit(training_set.drop(training_set.columns[-1], axis=1), classes)
    return logr

def test_error(logr, test_set):
    error = 1 - logr.score(test_set.drop(test_set.columns[-1], axis=1), test_set.iloc[: , -1])
    #print(error)
    return error

def test_logr(penalty, training_set, test_set):
    logr = penalty_logr(penalty, training_set)
    error = test_error(logr, test_set)
    return error

def get_best_penalty(training_set, test_set):
    penalty_list = [i for i in range(0, 4)]
    best_penalty = -1
    best_error = 1
    for penalty in penalty_list:
        error = test_logr(penalty, training_set, test_set)
        if (error < best_error):
            best_error = error
            best_penalty = penalty
    return best_penalty, best_error

def get_folds(data):
    folds = [data[0:int(0.2*len(data))], data[int(0.2*len(data)):int(0.4*len(data))], data[int(0.4*len(data)):int(0.6*len(data))], data[int(0.6*len(data)):int(0.8*len(data))], data[int(0.8*len(data)):int(1.0*len(data))]]
    return folds

def cross_validation(data):
    best_penalty = None
    best_error = 10
    folds = get_folds(data)
    for fold in folds:
        train_folds = []
        for f in folds:
            #print(len(f), f)
            if f is not fold:
                train_folds.append(f)
        train_set = pd.concat(train_folds)
        test_set = fold
        current_penalty, current_error = get_best_penalty(train_set, test_set)
        if current_error < best_error:
            best_penalty = current_penalty
            best_error = current_error
    penalty_number_used[best_penalty] += 1
    return best_penalty, best_error

def split_data_set(df_to_be_split: pd.DataFrame):
    holy_numbers = [i for i in range(0, len(df_to_be_split))]
    random.shuffle(holy_numbers)
    train_numbers = holy_numbers[:int(0.80*len(df_to_be_split))]
    test_numbers = holy_numbers[int(0.80*len(df_to_be_split)):int(1.00*len(df_to_be_split))]
    train_set = df_to_be_split.iloc[train_numbers]
    test_set = df_to_be_split.iloc[test_numbers]
    return train_set, test_set

# find best_penalty with cross validation
for i in range(iterations):
    data = [hearts, triangles, circles, squares]
    data_set = pd.concat(data)
    data_set = data_set.sample(frac=1)

    penalty, error = cross_validation(data_set)
    print(penalty, error)
    penalty_sum += penalty
    error_sum += error

print(penalty_number_used)
print(penalty_sum/iterations, error_sum/iterations)

error_sum = 0
best_penalty = -1
times_used = 0
for i in range(0, 4):
    if penalty_number_used[i] > times_used:
        best_penalty = i
        times_used = penalty_number_used[i]

# train final model on best_penalty found
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

    error = test_logr(best_penalty, data_train, data_test)
    error_sum += error

print(best_penalty, error_sum/iterations)