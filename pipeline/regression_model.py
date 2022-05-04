import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
)
from sklearn.model_selection import KFold
import pandas as pd

from pipeline.images_features import features_data_filter


def ctr_regression(X_train, y_train):
    """

    Args:
        X_train: traning data
        y_train:

    Returns: model

    """
    model = keras.Sequential()

    model.add(Flatten(input_shape=(np.shape(X_train[0], ))))

    model.add(Dense(units=106,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='tanh'
                    ))

    model.add(Dropout(0.5))

    model.add(Dense(units=26,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='sigmoid'
                    ))

    model.add(Dropout(0.2))

    model.add(Dense(units=34,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='relu'
                    ))
    model.add(Dense(units=1))

    adam = keras.optimizers.Adam(learning_rate=0.01, decay=0.0001)
    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=['mae', 'mse'])

    model.fit(X_train, y_train, epochs=10, batch_size=128)

    return model


def nReads_regression(X_train, y_train):
    """

    Args:
        X_train: traning data
        y_train:

    Returns: model

    """
    model = keras.Sequential()

    model.add(Flatten(input_shape=(np.shape(X_train[0], ))))

    model.add(Dense(units=90,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='tanh'
                    ))

    model.add(Dropout(0.1))

    model.add(Dense(units=18,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='relu'
                    ))

    model.add(Dropout(0.1))

    model.add(Dense(units=1))

    adam = keras.optimizers.Adam(learning_rate=0.01, decay=1e-05)
    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=['mae', 'mse'])

    model.fit(X_train, y_train, epochs=10, batch_size=128)

    return model


def nReads_ctr_regression(X_train, y_train):
    """

    Args:
        X_train: traning data
        y_train:

    Returns: model

    """
    model = keras.Sequential()

    model.add(Flatten(input_shape=(np.shape(X_train[0], ))))

    model.add(Dense(units=97,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='sigmoid'
                    ))

    model.add(Dropout(0.2))

    model.add(Dense(units=385,
                    bias_initializer='ones',
                    kernel_initializer='random_normal',
                    activation='sigmoid'
                    ))

    model.add(Dropout(0.8))

    model.add(Dense(units=2))

    adam = keras.optimizers.Adam(learning_rate=0.01, decay=1e-05)
    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=['mae', 'mse'])

    model.fit(X_train, y_train, epochs=10, batch_size=128)

    return model


def regression_prediction(X_train, y_train, X_test, model):
    """
    nReads and ctr prediction
    Args:
        X_train: training X value
        y_train: training y value
        X_test: x value to be predicted
        model: nReads and ctr predicted separately 'split'
               nReads and ctr predicted together 'merge'

    Returns: an output array, array[0] is nReads, array[1] is ctr
    """

    if model == 'split':
        model_pre = ctr_regression(X_train, y_train['ctr'])
        nReads_model = nReads_regression(X_train, y_train['normed_nReads'])
        y_ctr = model_pre.predict(X_test)
        y_nReads = nReads_model.predict(X_test)
        output = np.vstack((y_nReads.reshape(1, -1), y_ctr.reshape(1, -1)))

    if model == 'merge':
        model_pre = nReads_ctr_regression(X_train, y_train[['normed_nReads', 'ctr']])
        output = model.predict(X_test)

    return output, model_pre


def score_evaluation(y_pred, y_test, n):
    ctrList = y_pred.tolist()
    df_result = pd.DataFrame(ctrList, columns=['pre_ctr'])
    df_name = pd.DataFrame(y_test['iid'], columns=['iid']).reset_index(drop=True)
    df_pre = pd.concat([df_result, df_name], axis=1)
    df_pre_ = df_pre.sort_values("pre_ctr", ascending=False, ignore_index=True)[:n]

    df_ctr = pd.DataFrame(y_test['ctr'].values, columns=['true_ctr'])
    df_true = pd.concat([df_ctr, df_name], axis=1)
    df_true_ = df_true.sort_values("true_ctr", ascending=False, ignore_index=True)[:n]
    count = [x for x in df_true_["iid"].tolist() if x in df_pre_["iid"].tolist()]
    score = len(count) / n
    print("the score@", n, "is", score)

    return score


def regression_evaluation(X, y, model, dropout, k, n):
    """

    K-Fold cross validation to evaluate model
    Args:
         X: X vector
         y: class label
         model: nReads and ctr predicted separately 'split'
                nReads and ctr predicted together 'merge'
         k: k-Fold
         n: score @ n

    Returns : avarage cross validation score@n
    """
    kf = KFold(n_splits=k, shuffle=True)
    scoreList = []
    lenList = []
    y = y.reset_index(drop=True)
    for train_index, test_index in kf.split(X):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        if dropout == True:
            X_train, y_train = features_data_filter(X_train, y_train,
                                                    1, 0, 0)
            X_train = np.array(X_train)
            X_test = np.array(X_test)

        else:
            X_train = np.array(X_train)
            X_test = np.array(X_test)

        y_pred, model_pre = regression_prediction(X_train, y_train, X_test, model)
        # validationLoss.append(model_pre.evaluate(X_test, y_test['ctr']))
        score = score_evaluation(y_pred[1], y_test, n)
        scoreList.append(score)

        lenList.append(len(y_test))

    avg_score = sum(np.array(lenList) * np.array(scoreList)) / sum(lenList)
    # avg_mse = sum(np.array(validationLoss)) / sum(lenList)

    return avg_score
