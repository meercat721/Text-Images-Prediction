from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn import metrics
import string
import numpy as np
import pandas as pd


def get_labels(label_df, iidList):
    """
    get labels from label list with configurationId = 2
    Args:
        label_df:  label dataframe
        iidList: iid list from images features

    Returns: preprocessed label list

    """
    labelList = []
    for i in range(len(iidList)):
        arr = label_df[(label_df["imageId"] == int(iidList[i]))
                       & (label_df["configurationId"] == 2)]["labelValue"].values.flatten()
        punctuation_string = string.punctuation
        for i in punctuation_string:
            arr = str(arr).replace(i, '').replace('\n', '')
        labelList.append(arr)
    return labelList


def word2vec(X):
    """
    transform text to vector, by calculating word frequency
    Args:
        X: text list

    Returns: transformed X vector

    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X).toarray()
    return X


def classifier_prediction(X_bayes, y_bayes, X_test, model):
    """

    Args:
        X_bayes: vectorized X values
        y_bayes: 0-1 class
        X_test:  vectorized X values for prediction
        model:   BernoulliClassifier = 'bernoulli', MultinomialClassifier = 'multinomial', GaussianClassifier = 'gaussian'

    Returns:predicted class, exp of log probability

    """

    if model == 'bernoulli':
        bb = BernoulliNB().fit(X_bayes, y_bayes)
        y_pred = bb.predict(X_test)
        prob = bb.predict_log_proba(X_test)
        prob_df = pd.DataFrame(prob, columns=['zero', 'nonzero'])
        weights = prob_df.apply(lambda x: (np.exp(x)))

    if model == 'multinomial':
        mb = MultinomialNB().fit(X_bayes, y_bayes)
        y_pred = mb.predict(X_test)
        prob = mb.predict_log_proba(X_test)
        prob_df = pd.DataFrame(prob, columns=['zero', 'nonzero'])
        weights = prob_df.apply(lambda x: (np.exp(x)))

    if model == 'gaussian':
        gb = GaussianNB().fit(X_bayes, y_bayes)
        y_pred = gb.predict(X_test)
        prob = gb.predict_log_proba(X_test)
        prob_df = pd.DataFrame(prob, columns=['zero', 'nonzero'])
        weights = prob_df.apply(lambda x: (np.exp(x)))

    return y_pred, weights


def classifier_evaluation(X, y, model, k):
    """

    K-Fold cross validation to evaluate model
    Args:
         X: X vector
         y: class label
         model: BernoulliClassifier = 'bernoulli',
                     MultinomialClassifier = 'multinomial'
                     GaussianClassifier = 'gaussian'
         k: k-Fold

    Returns : cross validation accuracy, precision, recall
    """
    kf = KFold(n_splits=k, shuffle=True)
    acc_list = list()
    pre_list = list()
    rec_list = list()
    D = list()
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        y_pred, weights = classifier_prediction(X_train, y_train, X_test, model)
        acc = metrics.accuracy_score(y_test, y_pred)
        rec = metrics.recall_score(y_test, y_pred, average="macro")
        pre = metrics.precision_score(y_test, y_pred, average="macro")

        print('accuracy', acc, '\n',
              'precision', pre, '\n',
              'recall', rec)

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)

        D.append(len(y_test))

    acc = sum(np.array(D) * np.array(acc_list)) / sum(D)
    pre = sum(np.array(D) * np.array(pre_list)) / sum(D)
    rec = sum(np.array(D) * np.array(rec_list)) / sum(D)
    return acc, pre, rec
